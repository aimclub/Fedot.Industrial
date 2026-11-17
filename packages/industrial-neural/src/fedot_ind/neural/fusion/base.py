"""Abstract base class for FUTURE fusion strategies."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from collections.abc import Hashable


class BaseFusionStrategy(nn.Module, ABC):
    """Common helper methods for fusion strategy modules."""

    @staticmethod
    def _ordered_embeddings(
        embeddings: Mapping[Hashable, torch.Tensor],
        modalities: Sequence[Hashable],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(embeddings[modality] for modality in modalities)

    @abstractmethod
    def fuse(
        self,
        embeddings: Mapping[Hashable, torch.Tensor],
        modalities: Sequence[Hashable],
        *,
        raw_modality: Hashable | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Fuse modality embeddings through one public strategy boundary."""

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method.")
