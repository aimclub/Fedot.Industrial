"""Validation rules for FUTURE multimodal models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.future.enums import FusionMethod
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.rules import (
    normalize_unique_modalities,
    validate_bundle_type,
    validate_modalities_presence,
    validate_registry_supports_modalities,
)

__all__ = [
    "normalize_unique_modalities",
    "require_initialized_model_parts",
    "require_resolved_modalities",
    "validate_choice",
    "validate_context_modalities_for_raw_centered",
    "validate_divisible",
    "validate_embeddings_count",
    "validate_encoder_registry_has_modalities",
    "validate_modalities_presence",
    "validate_multimodal_bundle_input",
    "validate_positive_int",
    "validate_stacked_embeddings_shape",
    "validate_supported_fusion_method",
]


def validate_positive_int(name: str, value: int, min_value: int = 1) -> None:
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")


def validate_choice(name: str, value: Any, allowed: Sequence[Any]) -> None:
    if value not in allowed:
        options = ", ".join(repr(item) for item in allowed)
        raise ValueError(
            f"Unknown {name}={value!r}. Expected one of: {options}."
        )


def validate_divisible(
    dividend_name: str,
    dividend: int,
    divisor_name: str,
    divisor: int,
) -> None:
    if dividend % divisor != 0:
        raise ValueError(
            f"{dividend_name}={dividend} must be divisible by "
            f"{divisor_name}={divisor}."
        )


def validate_supported_fusion_method(
    fusion_method: FusionMethod | str,
    fusion_registry: Mapping[FusionMethod, Any],
) -> FusionMethod:
    if isinstance(fusion_method, FusionMethod):
        normalized = fusion_method
    else:
        try:
            normalized = FusionMethod(str(fusion_method))
        except ValueError as exc:
            available = [method.value for method in FusionMethod]
            raise ValueError(
                f"Unknown fusion method '{fusion_method}'. Available methods: {available}."
            ) from exc

    if normalized not in fusion_registry:
        available = [method.value for method in fusion_registry]
        raise ValueError(
            f"Unknown fusion method '{normalized.value}'. Available methods: {available}."
        )
    return normalized


def validate_multimodal_bundle_input(input_data: Any) -> MultimodalDataBundle:
    validate_bundle_type(input_data, MultimodalDataBundle)
    return input_data


def validate_encoder_registry_has_modalities(
    modalities: Sequence[MultimodalModality],
    preset_registry: Mapping[MultimodalModality, Any],
) -> None:
    validate_registry_supports_modalities(
        modalities=modalities,
        registry=preset_registry,
        registry_label="encoder presets",
    )


def validate_context_modalities_for_raw_centered(
    raw_modality: MultimodalModality,
    modalities: Sequence[MultimodalModality],
) -> None:
    if raw_modality not in modalities:
        raise ValueError(
            f"Raw-centered fusion requires raw modality '{raw_modality.value}' in modalities."
        )
    context_modalities = [modality for modality in modalities if modality != raw_modality]
    if len(context_modalities) == 0:
        raise ValueError("Raw-centered fusion requires at least one context modality.")


def require_resolved_modalities(
    modalities: Sequence[MultimodalModality] | None,
) -> Sequence[MultimodalModality]:
    """Return model modalities only after they have been resolved."""

    if modalities is None:
        raise ValueError("Model modalities are not resolved.")
    return modalities


def validate_embeddings_count(
    embeddings: tuple[torch.Tensor, ...],
    expected_count: int,
    label: str = "embeddings",
) -> None:
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {label}, got {len(embeddings)}."
        )


def validate_stacked_embeddings_shape(
    stacked_embeddings: torch.Tensor,
    expected_n_inputs: int,
    expected_d_model: int,
) -> None:
    _, n_inputs, d_model = stacked_embeddings.shape

    if n_inputs != expected_n_inputs:
        raise ValueError(
            f"Expected n_inputs={expected_n_inputs}, got {n_inputs}."
        )
    if d_model != expected_d_model:
        raise ValueError(
            f"Expected d_model={expected_d_model}, got {d_model}."
        )


def require_initialized_model_parts(
    encoders: nn.ModuleDict | None,
    fusion: nn.Module | None,
    modalities: Sequence[MultimodalModality] | None,
) -> tuple[nn.ModuleDict, nn.Module, Sequence[MultimodalModality]]:
    """Require initialized model parts and return normalized references."""

    modalities = require_resolved_modalities(modalities)

    if encoders is None:
        raise ValueError("Model encoders are not initialized.")
    if fusion is None:
        raise ValueError("Fusion module is not initialized.")
    return encoders, fusion, modalities
