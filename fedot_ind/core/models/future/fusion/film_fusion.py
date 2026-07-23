"""Feature-wise Linear Modulation (FiLM) fusion with raw as the main stream."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import torch
import torch.nn as nn

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.rules import (
    validate_positive_int,
    validate_embeddings_count,
)
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn
from fedot_ind.core.multimodal.enums import MultimodalModality


class FiLMFusion(BaseFusionStrategy):
    """Modulate a raw embedding with context-derived scale and shift.

    Context embeddings are concatenated, projected, and mapped to FiLM
    parameters ``gamma`` and ``beta``. The fused representation is:

    ``h_final = h_raw * (1 + gamma) + beta``

    With ``gamma=0`` and ``beta=0`` the output equals ``h_raw``, giving a
    stable raw-centered baseline at initialization.

    Forward
    -------
    h_raw : Tensor, shape ``(batch, d_model)``
        Main (raw) embedding to be modulated.
    *context_embeddings : Tensor
        ``n_context_inputs`` tensors, each of shape ``(batch, d_model)``.
    return_aux : bool
        If ``True``, return a dict with ``h_final``, ``gamma``, ``beta``, etc.
    """

    def __init__(
        self,
        n_context_inputs: int,
        d_model: int,
        context_hidden_dim: int = 128,
        film_hidden_dim: int = 128,
        dropout: float = 0.2,
        activation: str = "ReLU",
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_layernorm: bool = True,
    ):
        super().__init__()

        validate_positive_int(
            name="n_context_inputs",
            value=n_context_inputs,
            min_value=1,
        )

        self.n_context_inputs = n_context_inputs
        self.d_model = d_model
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

        context_in_dim = n_context_inputs * d_model

        context_layers: list[nn.Module] = [
            nn.Linear(context_in_dim, context_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, d_model),
        ]

        if use_layernorm:
            context_layers.append(nn.LayerNorm(d_model))

        self.context_projector = nn.Sequential(*context_layers)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(d_model, film_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, d_model),
            nn.Tanh(),
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(d_model, film_hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, d_model),
            nn.Tanh(),
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

        gamma = self.gamma_scale * self.gamma_mlp(h_context)
        beta = self.beta_scale * self.beta_mlp(h_context)
        h_final = h_raw * (1.0 + gamma) + beta

        if return_aux:
            return {
                "h_final": h_final,
                "h_raw": h_raw,
                "h_context": h_context,
                "gamma": gamma,
                "beta": beta,
            }

        return h_final

    def fuse(
        self,
        embeddings: Mapping[MultimodalModality, torch.Tensor],
        modalities: Sequence[MultimodalModality],
        *,
        raw_modality: MultimodalModality | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if raw_modality is None:
            raise ValueError("FiLM fusion requires raw_modality.")
        context_modalities = tuple(
            modality for modality in modalities if modality != raw_modality
        )
        h_raw = embeddings[raw_modality]
        context_embeddings = self._ordered_embeddings(embeddings, context_modalities)
        return self.forward(
            h_raw,
            *context_embeddings,
            return_aux=return_aux,
        )
