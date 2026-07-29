from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Sequence

from fedot_ind.core.multimodal.enums import (
    MultimodalModality,
    NormalizationConfig,
    NormalizationStep,
)
from fedot_ind.core.multimodal.mapping import (
    DEFAULT_STAT_FEATURE_CONFIG,
    DEFAULT_STAT_FEATURE_GLOBAL_CONFIG,
    DEFAULT_STAT_FEATURES as MAPPING_DEFAULT_STAT_FEATURES,
    MODALITY_CAPABILITIES,
    NORMALIZATION_HANDLERS,
)
from fedot_ind.core.multimodal.rules import (
    normalize_modality,
    validate_modalities_presence,
    validate_positive_number,
    validate_registry_supports_modalities,
    validate_registry_supports_normalization_steps,
)
from fedot_ind.core.operation.transformation.torch_backend.enums import StatisticalFeature


DEFAULT_STAT_FEATURES = tuple(MAPPING_DEFAULT_STAT_FEATURES)


def default_normalization_config() -> NormalizationConfig:
    return {
        MultimodalModality.stats: (
            NormalizationStep.imputation,
            NormalizationStep.feature_standardization,
        ),
        MultimodalModality.gaf: (NormalizationStep.image_standardization,),
        MultimodalModality.stft: (
            NormalizationStep.log1p,
            NormalizationStep.image_standardization,
        ),
    }


def default_transformation_config() -> dict[
    MultimodalModality,
    dict[str, Any],
]:
    return {
        MultimodalModality.raw: {
            "per_sample_z_normalize": False,
            "per_sample_z_normalize_eps": 1e-6,
        },
        MultimodalModality.stats: {
            "window_size": 12,
            "stride": 50,
            "add_global_features": True,
            "feature_names": DEFAULT_STAT_FEATURES,
            "stat_feature_config": DEFAULT_STAT_FEATURE_CONFIG,
            "stat_feature_global_config": DEFAULT_STAT_FEATURE_GLOBAL_CONFIG,
        },
        MultimodalModality.gaf: {
            "method": "summation",
            "overlapping": True,
            "image_size": 0.25,
            "sample_range": None,
        },
        MultimodalModality.stft: {
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


def normalization_policy_from_steps(steps: Sequence[NormalizationStep]) -> str:
    if not steps:
        return "none"
    if tuple(steps) == (
        NormalizationStep.imputation,
        NormalizationStep.feature_standardization,
    ):
        return "train_mean_imputation_then_train_mean_std"
    if tuple(steps) == (NormalizationStep.image_standardization,):
        return "train_image_standardization"
    if tuple(steps) == (
        NormalizationStep.log1p,
        NormalizationStep.image_standardization,
    ):
        return "log1p_then_train_image_standardization"
    return " -> ".join(step.value for step in steps)


@dataclass(frozen=True)
class PreparationConfig:
    """Normalized immutable contract for multimodal data preparation."""

    normalization_config: Mapping[
        MultimodalModality,
        tuple[NormalizationStep, ...],
    ]
    transformation_config: Mapping[
        MultimodalModality,
        Mapping[str, Any],
    ]
    torch_device: Any = "auto"
    preprocessor_eps: float = 1e-6
    auto_adjust_stft: bool = True

    @property
    def modalities(self) -> tuple[MultimodalModality, ...]:
        return tuple(self.transformation_config)

    def modality_config(self, modality: MultimodalModality) -> dict[str, Any]:
        return _mutable_copy(self.transformation_config.get(modality, {}))

    def stats_feature_names(self) -> tuple[str, ...]:
        return tuple(
            self.modality_config(MultimodalModality.stats).get(
                "feature_names",
                DEFAULT_STAT_FEATURES,
            )
        )

    def metadata(
        self,
        *,
        transform_params: Mapping[
            MultimodalModality,
            Mapping[str, Any],
        ] | None = None,
    ) -> dict[str, Any]:
        """Provenance for a prepared bundle.

        Device and dtype are deliberately absent: they are derived by
        MultimodalDataBundle from the tensors it holds.
        """

        resolved_transform_params = {
            modality: self.modality_config(modality)
            for modality in self.modalities
        }
        if transform_params is not None:
            resolved_transform_params.update(
                {
                    modality: dict(params)
                    for modality, params in transform_params.items()
                }
            )
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
                    else normalization_policy_from_steps(
                        self.normalization_config.get(modality, ())
                    )
                )
                for modality in self.modalities
            },
            "normalization_config": {
                modality.value: [
                    step.value
                    for step in self.normalization_config.get(modality, ())
                ]
                for modality in self.modalities
            },
            "transform_params": resolved_transform_params,
            "preparation_config": {
                "modalities": tuple(modality.value for modality in self.modalities),
                "transformation_config": {
                    modality.value: resolved_transform_params.get(modality, {})
                    for modality in self.modalities
                },
                "auto_adjust_stft": self.auto_adjust_stft,
            },
        }


def build_preparation_config(
    *,
    normalization_config: Mapping[
        MultimodalModality | str,
        Sequence[NormalizationStep | str],
    ]
    | None = None,
    transformation_config: Mapping[
        MultimodalModality | str,
        Mapping[str, Any],
    ]
    | None = None,
    torch_device: Any = "auto",
    preprocessor_eps: float = 1e-6,
    auto_adjust_stft: bool = True,
) -> PreparationConfig:
    """Normalize raw user input into an immutable PreparationConfig."""

    validate_positive_number("preprocessor_eps", preprocessor_eps)
    transformations = _normalize_transformation_config(transformation_config)
    normalization = _normalize_normalization_config(
        normalization_config,
        modalities=transformations,
    )
    return PreparationConfig(
        normalization_config=_freeze_mapping(normalization),
        transformation_config=_freeze_mapping(transformations),
        torch_device=torch_device,
        preprocessor_eps=preprocessor_eps,
        auto_adjust_stft=auto_adjust_stft,
    )


def _normalize_transformation_config(
    source: Mapping[MultimodalModality | str, Mapping[str, Any]] | None,
) -> dict[MultimodalModality, dict[str, Any]]:
    raw_source = default_transformation_config() if source is None else source
    normalized = {
        normalize_modality(modality): dict(params)
        for modality, params in raw_source.items()
    }
    validate_registry_supports_modalities(
        modalities=normalized,
        registry=MODALITY_CAPABILITIES,
        registry_label="preparation",
    )

    if MultimodalModality.raw not in normalized:
        normalized[MultimodalModality.raw] = dict(
            default_transformation_config()[MultimodalModality.raw]
        )

    raw_config = normalized[MultimodalModality.raw]
    eps = float(raw_config.get("per_sample_z_normalize_eps", 1e-6))
    if raw_config.get("per_sample_z_normalize", False):
        validate_positive_number("raw.per_sample_z_normalize_eps", eps)
    raw_config["per_sample_z_normalize_eps"] = eps

    if MultimodalModality.stats in normalized:
        stats_config = normalized[MultimodalModality.stats]
        features = stats_config.get("feature_names", DEFAULT_STAT_FEATURES)
        stats_config["feature_names"] = tuple(
            _normalize_stat_feature(feature) for feature in features
        )

    return {
        MultimodalModality.raw: normalized[MultimodalModality.raw],
        **{
            modality: params
            for modality, params in normalized.items()
            if modality is not MultimodalModality.raw
        },
    }


def _normalize_normalization_config(
    source: Mapping[
        MultimodalModality | str,
        Sequence[NormalizationStep | str],
    ]
    | None,
    *,
    modalities: Mapping[MultimodalModality, Any],
) -> dict[MultimodalModality, tuple[NormalizationStep, ...]]:
    if source is None:
        defaults = default_normalization_config()
        normalized = {
            modality: tuple(defaults[modality])
            for modality in modalities
            if modality in defaults
        }
    else:
        normalized = {
            normalize_modality(modality): tuple(
                _normalize_step(step) for step in steps
            )
            for modality, steps in source.items()
        }

    if MultimodalModality.raw in normalized:
        raise ValueError(
            "Raw modality is not normalized by MultimodalPreprocessor. "
            "Use transformation_config['raw']['per_sample_z_normalize']."
        )
    validate_modalities_presence(
        required=normalized,
        available=modalities,
        source_label="Transformation config",
    )
    validate_registry_supports_normalization_steps(
        normalized,
        NORMALIZATION_HANDLERS,
    )
    return normalized


def _normalize_step(value: NormalizationStep | str) -> NormalizationStep:
    if isinstance(value, NormalizationStep):
        return value
    return NormalizationStep(str(value))


def _normalize_stat_feature(value: StatisticalFeature | str) -> str:
    if isinstance(value, StatisticalFeature):
        return value.value
    return StatisticalFeature(str(value)).value


def _freeze_mapping(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(
        {key: _freeze_value(value) for key, value in mapping.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _mutable_copy(value: Any) -> Any:
    """Copy a frozen config value into mutable containers for transformers."""

    if isinstance(value, Mapping):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_mutable_copy(item) for item in value)
    return value
