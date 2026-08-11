"""Latent bottleneck encoder: cross-/self-attention over modality tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

from fedot_ind.core.models.future.rules import (
    validate_choice,
    validate_divisible,
    validate_embeddings_count,
    validate_positive_int,
    validate_stacked_embeddings_shape,
)


class BottleneckLatentBlock(nn.Module):
    """One bottleneck transformer block over latent and modality tokens.

    Pipeline (each step is residual):

    1. Cross-attention — latent tokens attend to modality tokens.
    2. Self-attention — latent tokens attend to each other.
    3. Feed-forward MLP on latent tokens.

    Forward
    -------
    latents : Tensor, shape ``(batch, num_latents, d_model)``
        Learnable bottleneck queries.
    modality_tokens : Tensor, shape ``(batch, num_modalities, d_model)``
        Stacked modality embeddings (with optional modality embeddings added).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.modality_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        modality_tokens: torch.Tensor,
        return_attn: bool = False,
    ):
        q = self.cross_attn_norm(latents)
        kv = self.modality_norm(modality_tokens)

        cross_out, cross_attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        latents = latents + cross_out

        qkv = self.self_attn_norm(latents)

        self_out, self_attn_weights = self.self_attn(
            query=qkv,
            key=qkv,
            value=qkv,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        latents = latents + self_out
        latents = latents + self.ffn(self.ffn_norm(latents))

        if return_attn:
            return latents, {
                "cross_attn": cross_attn_weights,
                "self_attn": self_attn_weights,
            }

        return latents


class BottleneckRepresentationEncoder(nn.Module):
    """Fuse modality embeddings through learnable latent bottleneck tokens.

    Steps:

    1. Stack ``n_modalities`` embeddings → modality tokens ``(B, M, D)``.
    2. Add learnable modality-type embeddings.
    3. Initialize ``num_latents`` learnable latent tokens ``(B, K, D)``.
    4. Apply :class:`BottleneckLatentBlock` layers (cross-attn + self-attn).
    5. Pool latents into a single vector ``(B, D)``.

    Pooling modes: ``"mean"`` (default), ``"cls"`` (first token),
    ``"concat"`` (flatten + projection).

    Forward
    -------
    *modality_embeddings : Tensor
        One tensor per modality, each of shape ``(batch, d_model)``.
    """

    def __init__(
        self,
        n_modalities: int,
        d_model: int,
        num_latents: int = 4,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "mean",
        latent_init_std: float = 0.02,
    ):
        super().__init__()

        validate_positive_int(name="n_modalities", value=n_modalities, min_value=1)
        validate_positive_int(name="num_latents", value=num_latents, min_value=1)
        validate_positive_int(name="num_layers", value=num_layers, min_value=1)
        validate_choice(
            name="pooling",
            value=pooling,
            allowed=("mean", "cls", "concat"),
        )
        validate_divisible(
            dividend_name="d_model",
            dividend=d_model,
            divisor_name="num_heads",
            divisor=num_heads,
        )

        self.n_modalities = n_modalities
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pooling = pooling

        self.modality_embedding = nn.Parameter(
            torch.zeros(1, n_modalities, d_model)
        )

        self.latent_tokens = nn.Parameter(
            torch.randn(num_latents, d_model) * latent_init_std
        )

        self.blocks = nn.ModuleList(
            [
                BottleneckLatentBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        if pooling == "concat":
            self.concat_projection = nn.Sequential(
                nn.Linear(num_latents * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.concat_projection = None

    def forward(
        self,
        *modality_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        validate_embeddings_count(
            embeddings=modality_embeddings,
            expected_count=self.n_modalities,
            label="modality embeddings",
        )

        modality_tokens = torch.stack(modality_embeddings, dim=1)
        validate_stacked_embeddings_shape(
            stacked_embeddings=modality_tokens,
            expected_n_inputs=self.n_modalities,
            expected_d_model=self.d_model,
        )

        batch_size = modality_tokens.shape[0]
        modality_tokens = modality_tokens + self.modality_embedding
        latents = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        all_attn: list[dict[str, torch.Tensor]] = []

        for block in self.blocks:
            if return_aux:
                latents, attn = block(
                    latents,
                    modality_tokens,
                    return_attn=True,
                )
                all_attn.append(attn)
            else:
                latents = block(
                    latents,
                    modality_tokens,
                    return_attn=False,
                )

        latents = self.final_norm(latents)

        if self.pooling == "mean":
            h_final = latents.mean(dim=1)
        elif self.pooling == "cls":
            h_final = latents[:, 0]
        elif self.pooling == "concat":
            h_flat = latents.reshape(batch_size, self.num_latents * self.d_model)
            h_final = self.concat_projection(h_flat)
        else:
            raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

        if return_aux:
            return {
                "h_final": h_final,
                "modality_tokens": modality_tokens,
                "latents": latents,
                "attn": all_attn,
                "pooling": self.pooling,
                "num_latents": self.num_latents,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            }

        return h_final
