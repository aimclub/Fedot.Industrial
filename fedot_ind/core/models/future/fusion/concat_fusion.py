"""Concatenation-based fusion of modality embeddings."""

import torch
import torch.nn as nn

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.rules import (
    validate_positive_int,
    validate_embeddings_count,
)
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn


class MultiConcatFusionMLP(BaseFusionStrategy):
    """Fuse modality embeddings by concatenation followed by an MLP.

    Computes ``MLP(concat(h_1, ..., h_n))`` where each ``h_i`` has shape
    ``(batch, d_model)``.

    Forward
    -------
    *embeddings : Tensor
        ``n_inputs`` tensors, each of shape ``(batch, d_model)``.
    Returns fused embedding of shape ``(batch, d_model)``.
    """

    def __init__(
        self,
        n_inputs: int,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "GELU",
    ):
        super().__init__()

        validate_positive_int(name="n_inputs", value=n_inputs, min_value=1)

        self.n_inputs = n_inputs
        self.d_model = d_model

        self.fusion = nn.Sequential(
            nn.Linear(n_inputs * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            get_activation_fn(activation),
            nn.Dropout(dropout),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        validate_embeddings_count(
            embeddings=embeddings,
            expected_count=self.n_inputs,
        )

        h = torch.cat(list(embeddings), dim=1)
        return self.fusion(h)
