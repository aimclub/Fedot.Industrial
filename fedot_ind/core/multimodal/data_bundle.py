from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional

import torch

from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.rules import (
    DERIVED_METADATA_KEYS,
    ModalitySpec,
    validate_metadata_without_derived_keys,
    validate_modalities_not_empty,
    validate_modality_axes,
    validate_modality_keys,
    validate_modality_ranks,
    validate_modality_sample_dimension,
    validate_modality_tensors,
    validate_target_sample_dimension,
    validate_target_sample_size,
    validate_target_type,
    validate_uniform_device,
    validate_uniform_dtype,
    validate_uniform_sample_size,
)

_UNSET: Any = object()


@dataclass(frozen=True, eq=False)
class MultimodalDataBundle:
    """Immutable container for multimodal time-series representations.

    Expected modalities:
        {
            "raw": torch.Tensor,
            "stats": torch.Tensor,
            "gaf": torch.Tensor,
            "stft": torch.Tensor,
        }

    The first dimension of each tensor is interpreted as the number of samples.

    State changes only through the ``with_*``/``replace`` methods, each of which
    returns a new validated bundle. ``metadata`` holds caller-supplied
    provenance merged with facts derived from the tensors themselves
    (``modalities``, ``shapes``, ``device``, ``dtype``); the derived facts are
    computed here and cannot be supplied from outside, so they can never
    contradict the data.

    ``specs`` is optional: a modality without a spec is only required to carry a
    sample dimension and to agree with the others on the number of samples.
    """

    modalities: Mapping[MultimodalModality, torch.Tensor]
    target: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    specs: Mapping[MultimodalModality, ModalitySpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", MappingProxyType(dict(self.modalities)))
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))

        validate_modalities_not_empty(self.modalities)
        validate_modality_keys(self.modalities)
        validate_modality_tensors(self.modalities)
        validate_modality_sample_dimension(self.modalities)
        validate_uniform_sample_size(self.modalities)
        validate_uniform_dtype(self.modalities)
        validate_uniform_device(self.modalities)
        validate_modality_ranks(self.modalities, self.specs)
        validate_modality_axes(self.modalities, self.specs)
        validate_target_type(self.target)
        validate_target_sample_dimension(self.target)
        validate_target_sample_size(self.target, self.n_samples)

        object.__setattr__(self, "metadata", self._normalized_metadata())

    @property
    def available_modalities(self) -> list[MultimodalModality]:
        return list(self.modalities.keys())

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.modalities.values())).shape[0])

    @property
    def shapes(self) -> dict[MultimodalModality, tuple[int, ...]]:
        return {name: tuple(tensor.shape) for name, tensor in self.modalities.items()}

    @property
    def device(self) -> torch.device:
        return next(iter(self.modalities.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.modalities.values())).dtype

    @property
    def user_metadata(self) -> dict[str, Any]:
        """Caller-supplied metadata without the facts derived from tensors."""

        return {
            key: value
            for key, value in self.metadata.items()
            if key not in DERIVED_METADATA_KEYS
        }

    def _derived_metadata(self) -> dict[str, Any]:
        return {
            "modalities": self.available_modalities,
            "shapes": self.shapes,
            "device": self.device,
            "dtype": self.dtype,
        }

    def _normalized_metadata(self) -> Mapping[str, Any]:
        """Single point where metadata is assembled."""

        metadata = dict(self.metadata)
        validate_metadata_without_derived_keys(metadata)
        metadata.setdefault("transform_params", {})
        metadata.update(self._derived_metadata())
        return MappingProxyType(metadata)

    def replace(
        self,
        *,
        modalities: Mapping[MultimodalModality, torch.Tensor] = _UNSET,
        target: Optional[torch.Tensor] = _UNSET,
        metadata: Mapping[str, Any] = _UNSET,
        specs: Mapping[MultimodalModality, ModalitySpec] = _UNSET,
    ) -> "MultimodalDataBundle":
        """Rebuild the bundle, keeping every argument left out untouched.

        ``target=None`` clears the target; omitting ``target`` keeps it.
        """

        return MultimodalDataBundle(
            modalities=self.modalities if modalities is _UNSET else modalities,
            target=self.target if target is _UNSET else target,
            metadata=self.user_metadata if metadata is _UNSET else metadata,
            specs=self.specs if specs is _UNSET else specs,
        )

    def with_modalities(
        self,
        modalities: Mapping[MultimodalModality, torch.Tensor],
    ) -> "MultimodalDataBundle":
        return self.replace(modalities=modalities)

    def with_target(self, target: torch.Tensor) -> "MultimodalDataBundle":
        return self.replace(target=target)

    def without_target(self) -> "MultimodalDataBundle":
        return self.replace(target=None)

    def with_metadata(self, **updates: Any) -> "MultimodalDataBundle":
        return self.replace(metadata={**self.user_metadata, **updates})

    def with_specs(
        self,
        specs: Mapping[MultimodalModality, ModalitySpec],
    ) -> "MultimodalDataBundle":
        return self.replace(specs=specs)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MultimodalDataBundle":
        """Move modalities to ``device``/``dtype``; the target only moves device."""

        if device is None and dtype is None:
            return self
        return self.replace(
            modalities={
                name: tensor.to(device=device, dtype=dtype)
                for name, tensor in self.modalities.items()
            },
            target=None if self.target is None else self.target.to(device=device),
        )
