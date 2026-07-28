"""Validation rules for multimodal bundles and modality configuration.

Conventions kept by this module:

* one function gates exactly one invariant, so a failure names the broken
  contract instead of a group of unrelated checks;
* whether a value is "supported" is decided by a registry passed in by the
  caller, never by a list duplicated here;
* modality coercion is defined once and imported everywhere else.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from fedot_ind.core.multimodal.enums import MultimodalModality

DERIVED_METADATA_KEYS = frozenset({"modalities", "shapes", "device", "dtype"})


def normalize_modality(modality: MultimodalModality | str) -> MultimodalModality:
    """Convert a user-provided modality value to MultimodalModality."""

    if isinstance(modality, MultimodalModality):
        return modality
    try:
        return MultimodalModality(str(modality))
    except ValueError as exc:
        supported = [item.value for item in MultimodalModality]
        raise ValueError(
            f"Unsupported modality {modality!r}. Supported values: {supported}."
        ) from exc


def normalize_unique_modalities(
    modalities: Sequence[MultimodalModality | str],
) -> tuple[MultimodalModality, ...]:
    """Coerce a modality sequence and reject empty or duplicated input."""

    validate_modalities_not_empty(modalities, source_label="Modality sequence")
    normalized = tuple(normalize_modality(modality) for modality in modalities)
    validate_no_duplicate_modalities(normalized)
    return normalized


@dataclass(frozen=True)
class ModalitySpec:
    """Optional rank and shape requirement for a single modality.

    ``shape`` describes the axes after the sample dimension; ``None`` marks an
    axis whose size is not constrained.
    """

    allowed_ndim: tuple[int, ...]
    shape: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        validate_spec_allowed_ndim_not_empty(self)
        validate_spec_allowed_ndim_keeps_sample_dimension(self)
        validate_spec_shape_matches_rank(self)


def validate_spec_allowed_ndim_not_empty(spec: ModalitySpec) -> None:
    if not spec.allowed_ndim:
        raise ValueError("ModalitySpec.allowed_ndim must list at least one rank.")


def validate_spec_allowed_ndim_keeps_sample_dimension(spec: ModalitySpec) -> None:
    if any(ndim < 1 for ndim in spec.allowed_ndim):
        raise ValueError(
            "ModalitySpec.allowed_ndim must be >= 1 to keep the sample dimension, "
            f"got {spec.allowed_ndim}."
        )


def validate_spec_shape_matches_rank(spec: ModalitySpec) -> None:
    if spec.shape is None:
        return
    if spec.allowed_ndim != (len(spec.shape) + 1,):
        raise ValueError(
            "ModalitySpec.shape describes the axes after the sample dimension, so "
            f"allowed_ndim must be ({len(spec.shape) + 1},), got {spec.allowed_ndim}."
        )


def validate_modalities_not_empty(
    modalities: Iterable[Any],
    source_label: str = "MultimodalDataBundle",
) -> None:
    if not tuple(modalities):
        raise ValueError(f"{source_label} requires at least one modality.")


def validate_no_duplicate_modalities(
    modalities: Sequence[MultimodalModality],
) -> None:
    counts = Counter(modalities)
    duplicated = sorted(
        modality.value for modality, count in counts.items() if count > 1
    )
    if duplicated:
        raise ValueError(
            "Duplicate modality entries are not allowed. "
            f"Duplicate modalities: {duplicated}."
        )


def validate_modality_keys(modalities: Mapping[Any, Any]) -> None:
    invalid = [key for key in modalities if not isinstance(key, MultimodalModality)]
    if invalid:
        raise TypeError(
            "Modality name must be MultimodalModality, got "
            f"{[type(key).__name__ for key in invalid]}."
        )


def validate_modality_tensors(modalities: Mapping[MultimodalModality, Any]) -> None:
    for name, tensor in modalities.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Modality '{name.value}' must be torch.Tensor, got {type(tensor)}."
            )


def validate_modality_sample_dimension(
    modalities: Mapping[MultimodalModality, torch.Tensor],
) -> None:
    for name, tensor in modalities.items():
        if tensor.ndim == 0:
            raise ValueError(
                f"Modality '{name.value}' must have sample dimension, "
                "but scalar tensor was provided."
            )


def validate_uniform_sample_size(
    modalities: Mapping[MultimodalModality, torch.Tensor],
) -> None:
    sample_sizes = {
        name.value: int(tensor.shape[0]) for name, tensor in modalities.items()
    }
    if len(set(sample_sizes.values())) != 1:
        raise ValueError(
            "All modalities must have the same number of samples. "
            f"Got sample sizes: {sample_sizes}."
        )


def validate_uniform_dtype(
    modalities: Mapping[MultimodalModality, torch.Tensor],
) -> None:
    dtypes = {name.value: tensor.dtype for name, tensor in modalities.items()}
    if len(set(dtypes.values())) != 1:
        raise ValueError(
            "All modalities must have the same dtype. "
            f"Got dtypes: {dtypes}."
        )


def validate_uniform_device(
    modalities: Mapping[MultimodalModality, torch.Tensor],
) -> None:
    devices = {name.value: tensor.device for name, tensor in modalities.items()}
    if len(set(devices.values())) != 1:
        raise ValueError(
            "All modalities must be on the same device. "
            f"Got devices: {devices}."
        )


def validate_modality_ranks(
    modalities: Mapping[MultimodalModality, torch.Tensor],
    specs: Mapping[MultimodalModality, ModalitySpec],
) -> None:
    for name, spec in specs.items():
        tensor = modalities.get(name)
        if tensor is None:
            continue
        if tensor.ndim not in spec.allowed_ndim:
            raise ValueError(
                f"Modality '{name.value}' must have rank in {spec.allowed_ndim}, "
                f"got rank {tensor.ndim} for shape {tuple(tensor.shape)}."
            )


def validate_modality_axes(
    modalities: Mapping[MultimodalModality, torch.Tensor],
    specs: Mapping[MultimodalModality, ModalitySpec],
) -> None:
    for name, spec in specs.items():
        tensor = modalities.get(name)
        if tensor is None or spec.shape is None:
            continue
        actual = tuple(tensor.shape[1:])
        mismatched = [
            axis
            for axis, expected in enumerate(spec.shape)
            if expected is not None and actual[axis] != expected
        ]
        if mismatched:
            raise ValueError(
                f"Modality '{name.value}' must have shape {spec.shape} after the "
                f"sample dimension, got {actual}. Mismatched axes: {mismatched}."
            )


def validate_target_type(target: Any) -> None:
    if target is None:
        return
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Target must be torch.Tensor or None, got {type(target)}.")


def validate_target_sample_dimension(target: torch.Tensor | None) -> None:
    if target is None:
        return
    if target.ndim == 0:
        raise ValueError("Target must have sample dimension.")


def validate_target_sample_size(target: torch.Tensor | None, n_samples: int) -> None:
    if target is None:
        return
    if int(target.shape[0]) != n_samples:
        raise ValueError(
            "Target and modalities must have the same number of samples. "
            f"Got target size {int(target.shape[0])}, modalities size {n_samples}."
        )


def validate_metadata_without_derived_keys(metadata: Mapping[str, Any]) -> None:
    conflicting = sorted(DERIVED_METADATA_KEYS.intersection(metadata))
    if conflicting:
        raise ValueError(
            "Derived metadata keys are computed from modality tensors and must not "
            f"be provided explicitly: {conflicting}. Use bundle.user_metadata to "
            "carry metadata over to a new bundle."
        )


def validate_modalities_presence(
    required: Iterable[MultimodalModality],
    available: Iterable[MultimodalModality],
    source_label: str,
) -> None:
    available = set(available)
    missing = sorted(
        modality.value for modality in required if modality not in available
    )
    if missing:
        raise ValueError(
            f"{source_label} does not contain required modalities: {missing}."
        )


def validate_registry_supports_modalities(
    modalities: Iterable[MultimodalModality],
    registry: Container[MultimodalModality],
    registry_label: str,
) -> None:
    unsupported = sorted(
        modality.value for modality in modalities if modality not in registry
    )
    if unsupported:
        raise ValueError(
            f"Unsupported modalities for {registry_label}: {unsupported}."
        )


def validate_registry_supports_normalization_steps(
    normalization_config: Mapping[MultimodalModality, Sequence[Any]],
    registry: Mapping[Any, Any],
) -> None:
    unsupported = sorted(
        {
            getattr(step, "value", str(step))
            for steps in normalization_config.values()
            for step in steps
            if step not in registry
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported normalization steps: {unsupported}.")


def validate_normalization_handler_types(
    registry: Mapping[Any, Any],
    base_class: type,
) -> None:
    invalid = sorted(
        getattr(step, "value", str(step))
        for step, handler_cls in registry.items()
        if not (isinstance(handler_cls, type) and issubclass(handler_cls, base_class))
    )
    if invalid:
        raise TypeError(
            f"Normalization handlers must inherit {base_class.__name__}: {invalid}."
        )


def validate_bundle_type(value: Any, bundle_type: type) -> None:
    if not isinstance(value, bundle_type):
        raise TypeError(
            f"Expected {bundle_type.__name__}, got {type(value).__name__}."
        )


def validate_positive_number(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
