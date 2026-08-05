"""Latent bottleneck fusion strategies over modality embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.fusion.bottleneck_encoder import (
    BottleneckRepresentationEncoder,
)
from fedot_ind.core.models.future.tools import summarize_bottleneck_attn
from fedot_ind.core.multimodal.enums import MultimodalModality


class BottleneckFusionBase(BaseFusionStrategy):
    """Shared residual / aux helpers for bottleneck fusion strategies."""

    def __init__(
        self,
        *,
        n_encoder_inputs: int,
        d_model: int,
        use_residual: bool,
        num_latents: int = 4,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "mean",
        latent_init_std: float = 0.02,
        alpha_is_vector: bool = True,
        residual_hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
        self.alpha_is_vector = alpha_is_vector

        self.encoder = BottleneckRepresentationEncoder(
            n_modalities=n_encoder_inputs,
            d_model=d_model,
            num_latents=num_latents,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pooling=pooling,
            latent_init_std=latent_init_std,
        )

        if use_residual:
            fusion_in_dim = 2 * d_model
            alpha_out_dim = d_model if alpha_is_vector else 1
            self.delta_mlp = nn.Sequential(
                nn.Linear(fusion_in_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(residual_hidden_dim, d_model),
            )
            self.alpha_mlp = nn.Sequential(
                nn.Linear(fusion_in_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(residual_hidden_dim, alpha_out_dim),
                nn.Sigmoid(),
            )
        else:
            self.delta_mlp = None
            self.alpha_mlp = None

    def _apply_residual(
        self,
        h_raw: torch.Tensor,
        h_bn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.delta_mlp is None or self.alpha_mlp is None:
            raise RuntimeError("Residual path is not configured.")
        fusion_input = torch.cat([h_raw, h_bn], dim=-1)
        delta = self.delta_mlp(fusion_input)
        alpha = self.alpha_mlp(fusion_input)
        h_final = h_raw + alpha * delta
        return h_final, delta, alpha

    def _build_aux(
        self,
        *,
        h_final: torch.Tensor,
        encoder_aux: Mapping[str, Any],
        h_raw: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        h_bn: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        aux: dict[str, Any] = {
            "h_final": h_final,
            "pooling": encoder_aux["pooling"],
            "num_latents": encoder_aux["num_latents"],
            "num_heads": encoder_aux["num_heads"],
            "num_layers": encoder_aux["num_layers"],
            "attention_summary": summarize_bottleneck_attn(encoder_aux["attn"]),
        }
        if h_raw is not None:
            aux["h_raw"] = h_raw
        if h_bn is not None:
            aux["h_context"] = h_bn
        if delta is not None:
            aux["delta"] = delta
        if alpha is not None:
            aux["alpha"] = alpha
        return aux

    def _forward_residual(
        self,
        h_raw: torch.Tensor,
        *encoder_inputs: torch.Tensor,
        return_aux: bool = False,
    ):
        if return_aux:
            encoder_aux = self.encoder(*encoder_inputs, return_aux=True)
            h_bn = encoder_aux["h_final"]
            h_final, delta, alpha = self._apply_residual(h_raw, h_bn)
            return self._build_aux(
                h_final=h_final,
                encoder_aux=encoder_aux,
                h_raw=h_raw,
                delta=delta,
                alpha=alpha,
                h_bn=h_bn,
            )

        h_bn = self.encoder(*encoder_inputs, return_aux=False)
        h_final, _, _ = self._apply_residual(h_raw, h_bn)
        return h_final


class OrdinaryBottleneckFusion(BottleneckFusionBase):
    """Ordinary latent bottleneck over all modality embeddings."""

    def __init__(
        self,
        n_inputs: int,
        d_model: int,
        **kwargs: Any,
    ):
        super().__init__(
            n_encoder_inputs=n_inputs,
            d_model=d_model,
            use_residual=False,
            **kwargs,
        )
        self.n_inputs = n_inputs

    def forward(
        self,
        *modality_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        if return_aux:
            encoder_aux = self.encoder(*modality_embeddings, return_aux=True)
            return self._build_aux(
                h_final=encoder_aux["h_final"],
                encoder_aux=encoder_aux,
            )
        return self.encoder(*modality_embeddings, return_aux=False)

    def fuse(
        self,
        embeddings: Mapping[MultimodalModality, torch.Tensor],
        modalities: Sequence[MultimodalModality],
        *,
        raw_modality: MultimodalModality | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        ordered = self._ordered_embeddings(embeddings, modalities)
        return self.forward(*ordered, return_aux=return_aux)


class RawResidualBottleneckFusion(BottleneckFusionBase):
    """Bottleneck over all modalities with a residual highway on raw."""

    def __init__(
        self,
        n_inputs: int,
        d_model: int,
        **kwargs: Any,
    ):
        super().__init__(
            n_encoder_inputs=n_inputs,
            d_model=d_model,
            use_residual=True,
            **kwargs,
        )
        self.n_inputs = n_inputs

    def forward(
        self,
        h_raw: torch.Tensor,
        *all_modality_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        # all_modality_embeddings must include h_raw in modality order.
        return self._forward_residual(
            h_raw,
            *all_modality_embeddings,
            return_aux=return_aux,
        )

    def fuse(
        self,
        embeddings: Mapping[MultimodalModality, torch.Tensor],
        modalities: Sequence[MultimodalModality],
        *,
        raw_modality: MultimodalModality | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        if raw_modality is None:
            raise ValueError("Raw residual bottleneck fusion requires raw_modality.")
        ordered = self._ordered_embeddings(embeddings, modalities)
        return self.forward(
            embeddings[raw_modality],
            *ordered,
            return_aux=return_aux,
        )


class ContextOnlyResidualBottleneckFusion(BottleneckFusionBase):
    """Context-only bottleneck with a residual highway on raw."""

    def __init__(
        self,
        n_context_inputs: int,
        d_model: int,
        **kwargs: Any,
    ):
        super().__init__(
            n_encoder_inputs=n_context_inputs,
            d_model=d_model,
            use_residual=True,
            **kwargs,
        )
        self.n_context_inputs = n_context_inputs

    def forward(
        self,
        h_raw: torch.Tensor,
        *context_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        return self._forward_residual(
            h_raw,
            *context_embeddings,
            return_aux=return_aux,
        )

    def fuse(
        self,
        embeddings: Mapping[MultimodalModality, torch.Tensor],
        modalities: Sequence[MultimodalModality],
        *,
        raw_modality: MultimodalModality | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        if raw_modality is None:
            raise ValueError(
                "Context-only residual bottleneck fusion requires raw_modality."
            )
        context_modalities = tuple(
            modality for modality in modalities if modality != raw_modality
        )
        context_embeddings = self._ordered_embeddings(embeddings, context_modalities)
        return self.forward(
            embeddings[raw_modality],
            *context_embeddings,
            return_aux=return_aux,
        )
