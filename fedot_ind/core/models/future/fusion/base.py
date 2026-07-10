"""Abstract base class for fusion strategies."""

import torch
import torch.nn as nn
from abc import ABC
from typing import Any


class BaseFusionStrategy(nn.Module, ABC):
    """Common helper methods for fusion strategy modules."""

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method.")
