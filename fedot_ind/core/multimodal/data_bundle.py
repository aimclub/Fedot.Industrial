from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from fedot_ind.core.multimodal.enums import MultimodalModality


@dataclass
class MultimodalDataBundle:
    """Container for multimodal time-series representations.

    Expected modalities:
        {
            "raw": torch.Tensor,
            "stats": torch.Tensor,
            "gaf": torch.Tensor,
            "stft": torch.Tensor,
        }

    The first dimension of each tensor is interpreted as the number of samples.
    """

    modalities: dict[MultimodalModality, torch.Tensor]
    target: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_modalities()
        self._validate_target()
        self.metadata = self._build_metadata(self.metadata)

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

    def _validate_modalities(self) -> None:
        if not self.modalities:
            raise ValueError("MultimodalDataBundle requires at least one modality.")

        for name, tensor in self.modalities.items():
            if not isinstance(name, MultimodalModality):
                raise TypeError(
                    f"Modality name must be MultimodalModality, got {type(name)}."
                )

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Modality '{name}' must be torch.Tensor, "
                    f"got {type(tensor)}."
                )

            if tensor.ndim == 0:
                raise ValueError(
                    f"Modality '{name}' must have sample dimension, "
                    "but scalar tensor was provided."
                )

        sample_sizes = {
            name: int(tensor.shape[0])
            for name, tensor in self.modalities.items()
        }

        unique_sample_sizes = set(sample_sizes.values())

        if len(unique_sample_sizes) != 1:
            raise ValueError(
                "All modalities must have the same number of samples. "
                f"Got sample sizes: {sample_sizes}."
            )

    def _validate_target(self) -> None:
        if self.target is None:
            return

        if not isinstance(self.target, torch.Tensor):
            raise TypeError(
                f"Target must be torch.Tensor or None, got {type(self.target)}."
            )

        if self.target.ndim == 0:
            raise ValueError("Target must have sample dimension.")

        if int(self.target.shape[0]) != self.n_samples:
            raise ValueError(
                "Target and modalities must have the same number of samples. "
                f"Got target size {int(self.target.shape[0])}, "
                f"modalities size {self.n_samples}."
            )

    def _build_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(metadata)

        metadata.setdefault("modalities", self.available_modalities)
        metadata.setdefault("shapes", self.shapes)
        metadata.setdefault("transform_params", {})
        metadata.setdefault("device", self.device)
        metadata.setdefault("dtype", self.dtype)

        return metadata

    def with_metadata(self, **updates: Any) -> "MultimodalDataBundle":
        metadata = dict(self.metadata)
        metadata.update(updates)
        return self.replace(metadata=metadata)

    def replace(
        self,
        *,
        modalities: dict[MultimodalModality, torch.Tensor] | None = None,
        target: Optional[torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
        keep_target: bool = True,
    ) -> "MultimodalDataBundle":
        resolved_target = self.target if keep_target else target
        if target is not None:
            resolved_target = target
        return MultimodalDataBundle(
            modalities=dict(self.modalities if modalities is None else modalities),
            target=resolved_target,
            metadata=dict(self.metadata if metadata is None else metadata),
        )
