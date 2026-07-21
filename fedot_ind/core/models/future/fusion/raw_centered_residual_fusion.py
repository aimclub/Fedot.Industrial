"""Raw-centered residual fusion: keep raw as the main representation."""

import torch
import torch.nn as nn

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.rules import (
    validate_positive_int,
    validate_embeddings_count,
)
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn


class RawCenteredResidualFusion(BaseFusionStrategy):
    """Add a gated context correction on top of the raw embedding.

    Context embeddings are projected and combined with ``h_raw`` to predict
    a residual ``delta`` and gate ``alpha``:

    ``h_final = h_raw + alpha * delta``

    When ``alpha -> 0`` the output stays close to ``h_raw``. Set
    ``alpha_is_vector=True`` (default) for per-dimension gating, or
    ``False`` for a single scalar gate.

    Forward
    -------
    h_raw : Tensor, shape ``(batch, d_model)``
        Main raw embedding (residual highway).
    *context_embeddings : Tensor
        ``n_context_inputs`` tensors, each of shape ``(batch, d_model)``.
    """

    def __init__(
        self,
        n_context_inputs: int,
        d_model: int,
        context_hidden_dim: int = 128,
        delta_hidden_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "ReLU",
        alpha_is_vector: bool = True,
    ):
        super().__init__()
        validate_positive_int(
            name="n_context_inputs",
            value=n_context_inputs,
            min_value=1,
        )

        self.n_context_inputs = n_context_inputs
        self.d_model = d_model
        self.alpha_is_vector = alpha_is_vector

        context_in_dim = n_context_inputs * d_model

        self.context_projector = nn.Sequential(
            nn.Linear(context_in_dim, context_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        fusion_in_dim = 2 * d_model

        self.delta_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, delta_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden_dim, d_model),
        )

        alpha_out_dim = d_model if alpha_is_vector else 1

        self.alpha_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, delta_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden_dim, alpha_out_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_raw: torch.Tensor,
        *context_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        validate_embeddings_count(
            embeddings=context_embeddings,
            expected_count=self.n_context_inputs,
            label="context embeddings",
        )

        h_context_cat = torch.cat(context_embeddings, dim=-1)
        h_context = self.context_projector(h_context_cat)
        fusion_input = torch.cat([h_raw, h_context], dim=-1)

        delta = self.delta_mlp(fusion_input)
        alpha = self.alpha_mlp(fusion_input)
        h_final = h_raw + alpha * delta

        if return_aux:
            return {
                "h_final": h_final,
                "h_raw": h_raw,
                "h_context": h_context,
                "delta": delta,
                "alpha": alpha,
            }

        return h_final
