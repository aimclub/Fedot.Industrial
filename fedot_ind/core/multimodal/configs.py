from dataclasses import dataclass
from typing import Any, Sequence

import torch

from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationConfig,
    NormalizationMethod,
    StatisticalFeature,
)
from fedot_ind.core.multimodal.mapping import (
    DEFAULT_STAT_FEATURES as MAPPING_DEFAULT_STAT_FEATURES,
    STAT_FEATURE_TO_METHOD,
)

DEFAULT_STAT_FEATURES = tuple(MAPPING_DEFAULT_STAT_FEATURES)
SUPPORTED_PREPARATION_MODALITIES = frozenset(
    (
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
        MultimodalModality.mtf,
    )
)


def default_normalization_config() -> NormalizationConfig:
    return {
        MultimodalModality.stats: (
            NormalizationMethod.imputation,
            NormalizationMethod.feature_standardization,
        ),
        MultimodalModality.gaf: (NormalizationMethod.image_standardization,),
        MultimodalModality.stft: (
            NormalizationMethod.log1p,
            NormalizationMethod.image_standardization,
        ),
    }


def default_transformation_config() -> dict[str, dict[str, Any]]:
    return {
        "raw": {
            "per_sample_z_normalize": False,
            "per_sample_z_normalize_eps": 1e-6,
        },
        "stats": {
            "window_size": 12,
            "stride": 50,
            "add_global_features": True,
            "feature_names": DEFAULT_STAT_FEATURES,
        },
        "gaf": {
            "method": "summation",
            "overlapping": True,
            "image_size": 0.25,
            "sample_range": None,
        },
        "stft": {
            "window_size": 64,
            "hop_length": 16,
            "n_fft": 64,
            "window_type": "hann",
            "center": False,
            "pad_mode": "reflect",
            "power": 2.0,
            "normalized": False,
        },
    }


def _normalization_policy(steps: Sequence[NormalizationMethod]) -> str:
    if not steps:
        return "none"
    if tuple(steps) == (
        NormalizationMethod.imputation,
        NormalizationMethod.feature_standardization,
    ):
        return "train_mean_imputation_then_train_mean_std"
    if tuple(steps) == (NormalizationMethod.image_standardization,):
        return "train_image_standardization"
    if tuple(steps) == (
        NormalizationMethod.log1p,
        NormalizationMethod.image_standardization,
    ):
        return "log1p_then_train_image_standardization"
    return " -> ".join(step.value for step in steps)


def _coerce_modality(value: MultimodalModality | str) -> MultimodalModality:
    if isinstance(value, MultimodalModality):
        return value
    try:
        return MultimodalModality(str(value))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported modality key {value!r}. "
            f"Supported values: {[modality.value for modality in MultimodalModality]}."
        ) from exc


def _coerce_stat_feature(value: StatisticalFeature | str) -> str:
    if isinstance(value, StatisticalFeature):
        return value.value
    return StatisticalFeature(str(value)).value


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for benchmark/DataLoader multimodal representation building."""

    normalization_config: NormalizationConfig | None = None
    transformation_config: dict[
        MultimodalModality | str,
        dict[str, Any],
    ] | None = None
    torch_device: Any = "auto"
    preprocessor_eps: float = 1e-6
    auto_adjust_stft: bool = True

    def __post_init__(self) -> None:
        transformation_source = (
            default_transformation_config()
            if self.transformation_config is None
            else self.transformation_config
        )
        normalized_transform_config: dict[MultimodalModality, dict[str, Any]] = {}
        for key, params in transformation_source.items():
            modality = _coerce_modality(key)
            if modality not in SUPPORTED_PREPARATION_MODALITIES:
                raise ValueError(f"Unsupported preparation modality: {modality.value}.")
            normalized_transform_config[modality] = dict(params)

        if MultimodalModality.raw not in normalized_transform_config:
            normalized_transform_config[MultimodalModality.raw] = dict(
                default_transformation_config()["raw"]
            )

        raw_config = normalized_transform_config[MultimodalModality.raw]
        if raw_config.get("per_sample_z_normalize", False):
            eps = float(raw_config.get("per_sample_z_normalize_eps", 1e-6))
            if eps <= 0:
                raise ValueError("raw.per_sample_z_normalize_eps must be positive.")
            raw_config["per_sample_z_normalize_eps"] = eps
        else:
            raw_config.setdefault("per_sample_z_normalize_eps", 1e-6)

        stats_config = normalized_transform_config.get(MultimodalModality.stats, {})
        stats_features = stats_config.get("feature_names", DEFAULT_STAT_FEATURES)
        stats_config["feature_names"] = tuple(
            _coerce_stat_feature(feature) for feature in stats_features
        )
        unsupported = sorted(set(stats_config["feature_names"]) - set(STAT_FEATURE_TO_METHOD))
        if unsupported:
            raise ValueError(f"Unsupported statistical features: {unsupported}.")
        if MultimodalModality.stats in normalized_transform_config:
            normalized_transform_config[MultimodalModality.stats] = stats_config

        modalities = (MultimodalModality.raw,) + tuple(
            modality
            for modality in normalized_transform_config
            if modality is not MultimodalModality.raw
        )

        if self.normalization_config is None:
            defaults = default_normalization_config()
            normalization_config = {
                modality: defaults[modality]
                for modality in modalities
                if modality in defaults
            }
        else:
            normalization_config = {
                _coerce_modality(modality): tuple(steps)
                for modality, steps in self.normalization_config.items()
            }
        if MultimodalModality.raw in normalization_config:
            raise ValueError(
                "Raw modality is not normalized by MultimodalPreprocessor. "
                "Use transformation_config['raw']['per_sample_z_normalize']."
            )
        object.__setattr__(self, "transformation_config", normalized_transform_config)
        object.__setattr__(self, "normalization_config", normalization_config)

    @property
    def modalities(self) -> tuple[MultimodalModality, ...]:
        config = self.transformation_config or {}
        return (MultimodalModality.raw,) + tuple(
            modality
            for modality in config
            if modality is not MultimodalModality.raw
        )

    def modality_config(self, modality: MultimodalModality) -> dict[str, Any]:
        config = self.transformation_config or {}
        return dict(config.get(modality, {}))

    def stats_feature_names(self) -> tuple[str, ...]:
        return tuple(
            self.modality_config(MultimodalModality.stats).get(
                "feature_names",
                DEFAULT_STAT_FEATURES,
            )
        )

    def metadata(self, device: torch.device) -> dict[str, Any]:
        normalization_config = self.normalization_config or {}
        transform_params = {
            modality: self.modality_config(modality)
            for modality in self.modalities
        }
        return {
            "normalization": {
                modality: (
                    "per_sample_z_norm"
                    if modality is MultimodalModality.raw
                    and bool(
                        self.modality_config(MultimodalModality.raw).get(
                            "per_sample_z_normalize",
                            False,
                        )
                    )
                    else _normalization_policy(
                        tuple(normalization_config.get(modality, ()))
                    )
                )
                for modality in self.modalities
            },
            "normalization_config": {
                modality.value: [
                    step.value
                    for step in normalization_config.get(modality, ())
                ]
                for modality in self.modalities
            },
            "transform_params": transform_params,
            "preparation_config": {
                "modalities": tuple(modality.value for modality in self.modalities),
                "transformation_config": {
                    modality.value: transform_params.get(modality, {})
                    for modality in self.modalities
                },
                "auto_adjust_stft": self.auto_adjust_stft,
            },
            "device": device,
            "dtype": torch.float32,
        }
