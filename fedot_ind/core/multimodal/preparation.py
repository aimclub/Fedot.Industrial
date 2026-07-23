from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.preprocessor import MultimodalPreprocessor
from fedot_ind.core.multimodal.configs import PreparationConfig
from fedot_ind.core.multimodal.mapping import (
    DEFAULT_STAT_FEATURE_CONFIG,
    DEFAULT_STAT_FEATURE_GLOBAL_CONFIG,
    TRANSFORMATION_HANDLERS,
)
from fedot_ind.core.operation.transformation.torch_backend.enums import StatisticalFeature
from fedot_ind.core.operation.transformation.torch_backend.io import (
    normalize_time_series_tensor,
    resolve_torch_device,
)
from fedot_ind.core.operation.transformation.torch_backend.statistical.tools import (
    normalize_feature_key,
)


def per_sample_z_normalize(series: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = series.mean(dim=-1, keepdim=True)
    std = series.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return torch.nan_to_num((series - mean) / std)


@dataclass
class MultimodalDatasetPreparer:
    """Build and normalize multimodal time-series bundles from Industrial inputs."""

    config: PreparationConfig = field(default_factory=PreparationConfig)
    preprocessor_: MultimodalPreprocessor | None = field(default=None, init=False)
    resolved_torch_device_: torch.device | None = field(default=None, init=False)
    label_mapping_: dict[Any, int] | None = field(default=None, init=False)

    def fit(self, X: Any, y: Any | None = None) -> "MultimodalDatasetPreparer":
        bundle = self._build_bundle(X, y, fit_target=True)
        self.preprocessor_ = MultimodalPreprocessor(
            normalization_config=self.config.normalization_config,
            eps=self.config.preprocessor_eps,
        ).fit(bundle)
        return self

    def transform(self, X: Any, y: Any | None = None) -> MultimodalDataBundle:
        if self.preprocessor_ is None:
            raise ValueError("MultimodalDatasetPreparer must be fitted before transform.")
        bundle = self._build_bundle(X, y, fit_target=False)
        return self.preprocessor_.transform(bundle)

    def fit_transform(self, X: Any, y: Any | None = None) -> MultimodalDataBundle:
        bundle = self._build_bundle(X, y, fit_target=True)
        self.preprocessor_ = MultimodalPreprocessor(
            normalization_config=self.config.normalization_config,
            eps=self.config.preprocessor_eps,
        ).fit(bundle)
        return self.preprocessor_.transform(bundle)

    def prepare_train_test(
        self,
        train_data: tuple[Any, Any],
        test_data: tuple[Any, Any],
    ) -> tuple[MultimodalDataBundle, MultimodalDataBundle]:
        train_features, train_target = train_data
        test_features, test_target = test_data
        train_bundle = self.fit_transform(train_features, train_target)
        test_bundle = self.transform(test_features, test_target)
        return train_bundle, test_bundle

    def prepare_classification_record(
        self,
        record: Any,
    ) -> tuple[MultimodalDataBundle, MultimodalDataBundle]:
        train_bundle, test_bundle = self.prepare_train_test(
            (record.train_features, record.train_target),
            (record.test_features, record.test_target),
        )
        train_bundle = self._with_source_metadata(
            train_bundle,
            {
                "kind": "ClassificationDatasetRecord",
                "benchmark": getattr(record, "benchmark", None),
                "dataset_name": getattr(record, "dataset_name", None),
                "subset": getattr(record, "subset", None),
                "split": "train",
            },
        )
        test_bundle = self._with_source_metadata(
            test_bundle,
            {
                "kind": "ClassificationDatasetRecord",
                "benchmark": getattr(record, "benchmark", None),
                "dataset_name": getattr(record, "dataset_name", None),
                "subset": getattr(record, "subset", None),
                "split": "test",
            },
        )
        return train_bundle, test_bundle

    def prepare_from_loader(
        self,
        loader: Any,
    ) -> tuple[MultimodalDataBundle, MultimodalDataBundle]:
        if not hasattr(loader, "load_data"):
            raise TypeError("loader must provide a load_data() method.")
        train_data, test_data = loader.load_data()
        train_bundle, test_bundle = self.prepare_train_test(train_data, test_data)
        train_bundle = self._with_source_metadata(
            train_bundle,
            {
                "kind": "DataLoader",
                "dataset_name": getattr(loader, "dataset_name", None),
                "split": "train",
            },
        )
        test_bundle = self._with_source_metadata(
            test_bundle,
            {
                "kind": "DataLoader",
                "dataset_name": getattr(loader, "dataset_name", None),
                "split": "test",
            },
        )
        return train_bundle, test_bundle

    @staticmethod
    def _with_source_metadata(
        bundle: MultimodalDataBundle,
        source_updates: dict[str, Any],
    ) -> MultimodalDataBundle:
        source = dict(bundle.metadata.get("source", {}))
        source.update(source_updates)
        return bundle.with_metadata(source=source)

    def _build_bundle(
        self,
        X: Any,
        y: Any | None,
        *,
        fit_target: bool,
    ) -> MultimodalDataBundle:
        device = self._resolve_device()
        series = torch.as_tensor(
            normalize_time_series_tensor(X),
            dtype=torch.float32,
            device=device,
        )
        raw_config = self.config.modality_config(MultimodalModality.raw)
        if raw_config.get("per_sample_z_normalize", False):
            series = per_sample_z_normalize(
                series,
                float(raw_config.get("per_sample_z_normalize_eps", 1e-6)),
            )

        modalities: dict[MultimodalModality, torch.Tensor] = {}
        resolved_transform_params: dict[MultimodalModality, dict[str, Any]] = {}
        for modality in self.config.modalities:
            tensor, params = self._build_modality(modality, series)
            modalities[modality] = tensor
            resolved_transform_params[modality] = params

        target, target_metadata = self._target_to_tensor(y, device, fit_target=fit_target)
        metadata = self.config.metadata(
            device,
            transform_params=resolved_transform_params,
        )
        metadata.update(target_metadata)
        return MultimodalDataBundle(
            modalities=modalities,
            target=target,
            metadata=metadata,
        )

    def _build_modality(
        self,
        modality: MultimodalModality,
        series: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if modality is MultimodalModality.raw:
            return series, self.config.modality_config(MultimodalModality.raw)
        params = self._resolve_modality_params(modality, series.shape[-1])
        params["torch_device"] = self._resolve_device()
        transformer_cls = TRANSFORMATION_HANDLERS.get(modality)
        if transformer_cls is None:
            raise ValueError(f"Unsupported modality: {modality}.")
        transformer = transformer_cls(params)
        return transformer.transform(series), params

    @staticmethod
    def _normalize_stats_config(config: Any) -> dict[StatisticalFeature, dict[str, Any]]:
        normalized: dict[StatisticalFeature, dict[str, Any]] = {}
        if not config:
            return normalized
        for raw_key, raw_kwargs in dict(config).items():
            key = normalize_feature_key(raw_key)
            normalized[key] = dict(raw_kwargs or {})
        return normalized

    @staticmethod
    def _feature_configs_from_names(
        feature_names: Any,
    ) -> tuple[dict[StatisticalFeature, dict[str, Any]], dict[StatisticalFeature, dict[str, Any]]]:
        local = {}
        global_ = {}
        local_methods = set(DEFAULT_STAT_FEATURE_CONFIG)
        global_methods = set(DEFAULT_STAT_FEATURE_GLOBAL_CONFIG)
        for feature_name in feature_names:
            feature = normalize_feature_key(feature_name)
            if feature in local_methods:
                local[feature] = {}
            elif feature in global_methods:
                global_[feature] = {}
            else:
                raise ValueError(f"Unsupported statistical feature: {feature_name}")
        if StatisticalFeature.n_peaks in global_:
            global_[StatisticalFeature.n_peaks]["normalized"] = True
        if StatisticalFeature.mean_ptp_distance in global_:
            global_[StatisticalFeature.mean_ptp_distance]["normalized"] = True
        return local, global_

    def _resolve_modality_params(
        self,
        modality: MultimodalModality,
        n_timestamps: int,
    ) -> dict[str, Any]:
        if modality is MultimodalModality.stats:
            return self._resolve_stats_params()
        if modality is MultimodalModality.gaf:
            return self._resolve_gaf_params(n_timestamps)
        if modality is MultimodalModality.stft:
            return self._resolve_stft_params(n_timestamps)
        return self.config.modality_config(modality)

    def _resolve_stats_params(self) -> dict[str, Any]:
        params = self.config.modality_config(MultimodalModality.stats)
        local_config = self._normalize_stats_config(
            params.get("stat_feature_config", DEFAULT_STAT_FEATURE_CONFIG)
        )
        global_config = self._normalize_stats_config(
            params.get("stat_feature_global_config", DEFAULT_STAT_FEATURE_GLOBAL_CONFIG)
        )

        feature_names = params.get("feature_names")
        if feature_names is not None:
            local_config, global_config = self._feature_configs_from_names(feature_names)

        params["stat_feature_config"] = local_config
        params["stat_feature_global_config"] = global_config
        params["add_global_features"] = bool(params.get("add_global_features", True)) and bool(
            global_config
        )
        return params

    def _resolve_gaf_params(self, n_timestamps: int) -> dict[str, Any]:
        if n_timestamps < 2:
            raise ValueError(
                f"GAF modality requires series length >= 2, got {n_timestamps}."
            )
        params = self.config.modality_config(MultimodalModality.gaf)
        window_size = params.get("window_size", None)
        if window_size is None:
            image_size = params.get("image_size", 1.0)
            if isinstance(image_size, float):
                n_segments = math.ceil(float(image_size) * n_timestamps)
            else:
                n_segments = int(image_size)
        else:
            n_segments = math.ceil(n_timestamps / max(1, int(window_size)))
        if n_segments < 2:
            # GAF implementation requires at least two PAA segments.
            params.pop("window_size", None)
            params["image_size"] = 1.0
            n_segments = 2
        if bool(params.get("overlapping", False)) and n_segments < 2:
            params["overlapping"] = False
        if n_segments == 2:
            params["overlapping"] = False
        return params

    def _resolve_stft_params(self, n_timestamps: int) -> dict[str, Any]:
        if n_timestamps < 2:
            raise ValueError(
                f"STFT modality requires series length >= 2, got {n_timestamps}."
            )
        params = self.config.modality_config(MultimodalModality.stft)
        if not self.config.auto_adjust_stft:
            return params

        configured_window = int(params.get("window_size", n_timestamps))
        window_size = max(1, min(configured_window, n_timestamps))
        configured_n_fft = int(params.get("n_fft", window_size))
        n_fft = max(window_size, min(configured_n_fft, n_timestamps))
        configured_hop = int(params.get("hop_length", max(1, window_size // 2)))
        hop_length = max(1, min(configured_hop, window_size))
        params.update(
            {
                "window_size": window_size,
                "n_fft": n_fft,
                "hop_length": hop_length,
            }
        )
        return params

    def _target_to_tensor(
        self,
        y: Any | None,
        device: torch.device,
        *,
        fit_target: bool,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if y is None:
            return None, {}

        values = np.asarray(y).reshape(-1)
        if values.dtype.kind in {"b", "i", "u"}:
            return torch.as_tensor(values, dtype=torch.long, device=device), {}
        if values.dtype.kind == "f":
            return torch.as_tensor(values, dtype=torch.float32, device=device), {}

        labels = tuple(str(label) for label in values.tolist())
        if fit_target or self.label_mapping_ is None:
            self.label_mapping_ = {
                label: index
                for index, label in enumerate(sorted(set(labels)))
            }
        unknown = sorted(set(labels) - set(self.label_mapping_))
        if unknown:
            raise ValueError(f"Unknown target labels during transform: {unknown}.")
        encoded = [self.label_mapping_[label] for label in labels]
        return (
            torch.as_tensor(encoded, dtype=torch.long, device=device),
            {"target_labels": tuple(self.label_mapping_.keys())},
        )

    def _resolve_device(self) -> torch.device:
        if self.resolved_torch_device_ is None:
            self.resolved_torch_device_ = resolve_torch_device(self.config.torch_device)
        return self.resolved_torch_device_
