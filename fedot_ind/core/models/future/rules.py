"""Validation rules for FUTURE multimodal models."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.future.mapping import FusionMethod
from fedot_ind.core.models.nn.models_rules import normalize_modality
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


def validate_positive_int(name: str, value: int, min_value: int = 1) -> None:
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")


def validate_non_empty_modalities(
    modalities: Sequence[MultimodalModality | str],
) -> None:
    if len(modalities) == 0:
        raise ValueError("modalities must contain at least one modality.")


def normalize_unique_modalities(
    modalities: Sequence[MultimodalModality | str],
) -> tuple[MultimodalModality, ...]:
    validate_non_empty_modalities(modalities)
    normalized: list[MultimodalModality] = []
    seen: set[MultimodalModality] = set()
    for modality in modalities:
        normalized_modality = normalize_modality(modality)
        if normalized_modality in seen:
            raise ValueError(
                f"Duplicate modalities are not allowed. Got: {tuple(modalities)}."
            )
        seen.add(normalized_modality)
        normalized.append(normalized_modality)
    return tuple(normalized)


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
    if not isinstance(input_data, MultimodalDataBundle):
        raise ValueError(
            f"Expected input to be MultimodalDataBundle, got {type(input_data)}."
        )
    return input_data


def resolve_modalities_from_bundle(
    bundle: MultimodalDataBundle,
    modalities: Sequence[MultimodalModality | str] | None = None,
) -> tuple[MultimodalModality, ...]:
    if modalities is None:
        resolved = tuple(bundle.metadata.get("modalities", bundle.available_modalities))
    else:
        resolved = tuple(modalities)
    return normalize_unique_modalities(resolved)


def validate_modalities_presence(
    required_modalities: Sequence[MultimodalModality],
    available_modalities: Sequence[MultimodalModality],
    source_label: str,
) -> None:
    missing = [
        modality.value
        for modality in required_modalities
        if modality not in available_modalities
    ]
    if missing:
        raise ValueError(
            f"{source_label} does not contain required modalities: {sorted(missing)}."
        )


def validate_encoder_registry_has_modalities(
    modalities: Sequence[MultimodalModality],
    preset_registry: Mapping[MultimodalModality, Any],
) -> None:
    unsupported = [modality.value for modality in modalities if modality not in preset_registry]
    if unsupported:
        raise ValueError(
            f"Unsupported modalities for encoder presets: {sorted(unsupported)}."
        )


def validate_supported_modalities(modalities: Sequence[MultimodalModality]) -> None:
    supported_modalities = frozenset(MultimodalModality)
    unsupported = [m.value for m in modalities if m not in supported_modalities]
    if unsupported:
        raise ValueError(
            "Supported modalities are "
            f"{sorted(m.value for m in supported_modalities)}. "
            f"Got unsupported modalities: {sorted(unsupported)}."
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
    require_modules: bool = True,
) -> tuple[nn.ModuleDict | None, nn.Module | None, Sequence[MultimodalModality]]:
    """Require initialized model parts and return normalized references."""

    if modalities is None:
        raise ValueError("Model modalities are not resolved.")

    if require_modules and encoders is None:
        raise ValueError("Model encoders are not initialized.")
    if require_modules and fusion is None:
        raise ValueError("Fusion module is not initialized.")
    return encoders, fusion, modalities
