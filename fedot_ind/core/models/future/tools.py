"""Tools for FUTURE models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Mapping
from typing import Optional

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@dataclass
class FutureTrainingConfig:
    """Hyperparameters for FUTURE classifier training."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = None
    device: Any = "cpu"
    seed: int | None = 42


@dataclass
class FutureTrainingHistory:
    """Lifecycle diagnostics produced by FUTURE classifier training."""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float | None] = field(default_factory=list)
    best_epoch: int = 0
    best_validation_loss: float | None = None
    train_duration_s: float = 0.0
    stopped_early: bool = False
    num_parameters: int | None = None


def summarize_bottleneck_attn(
    attn_list: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compact cross/self-attention diagnostics over all bottleneck layers."""

    if not attn_list:
        return {}

    cross_means: list[float] = []
    cross_stds: list[float] = []
    self_means: list[float] = []
    self_stds: list[float] = []

    for layer_attn in attn_list:
        cross = layer_attn.get("cross_attn")
        self_attn = layer_attn.get("self_attn")
        if cross is not None:
            detached = cross.detach()
            cross_means.append(float(detached.mean().item()))
            cross_stds.append(float(detached.std(unbiased=False).item()))
        if self_attn is not None:
            detached = self_attn.detach()
            self_means.append(float(detached.mean().item()))
            self_stds.append(float(detached.std(unbiased=False).item()))

    summary: dict[str, float] = {"num_layers_with_attn": float(len(attn_list))}
    if cross_means:
        summary["cross_attn_mean"] = sum(cross_means) / len(cross_means)
        summary["cross_attn_std"] = sum(cross_stds) / len(cross_stds)
    if self_means:
        summary["self_attn_mean"] = sum(self_means) / len(self_means)
        summary["self_attn_std"] = sum(self_stds) / len(self_stds)
    return summary


@dataclass(frozen=True)
class AuxOutputConfig:
    """Configuration for auxiliary diagnostics payload."""

    include_embeddings: bool = True
    include_num_parameters: bool = True
    include_fusion_aux: bool = True


KNOWN_FUSION_KEYS = frozenset(
    {
        "gates",
        "alpha",
        "gamma",
        "beta",
        "h_raw",
        "h_context",
        "delta",
        "h_final",
        "attention_summary",
        "pooling",
        "num_latents",
        "num_heads",
        "num_layers",
    }
)


@dataclass
class FusionAuxOutput:
    """Typed auxiliary output for multimodal fusion classifier."""

    logits: torch.Tensor
    h_final: torch.Tensor
    active_modalities: list[str]
    embedding_dim: int
    num_parameters: Optional[dict[str, Any]] = None
    embeddings: Optional[dict[str, torch.Tensor]] = None
    gates: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    gamma: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    h_raw: Optional[torch.Tensor] = None
    h_context: Optional[torch.Tensor] = None
    delta: Optional[torch.Tensor] = None
    alpha_stats: Optional[dict[str, float]] = None
    gamma_beta_summary: Optional[dict[str, float]] = None
    attention_summary: Optional[dict[str, float]] = None
    pooling: Optional[str] = None
    num_latents: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    extra: Optional[dict[str, Any]] = None

    @staticmethod
    def _summary_stats(tensor: torch.Tensor) -> dict[str, float]:
        detached = tensor.detach()
        return {
            "mean": float(detached.mean().item()),
            "std": float(detached.std(unbiased=False).item()),
            "l2_norm": float(detached.norm(p=2).item()),
        }

    def populate_fusion(
        self,
        fusion_aux: Mapping[str, Any],
        include_fusion_aux: bool,
        known_fusion_keys: set[str] | frozenset[str] = KNOWN_FUSION_KEYS,
    ) -> None:
        """Populate fusion-specific diagnostics in-place."""
        if not include_fusion_aux:
            return

        self.gates = fusion_aux.get("gates")
        self.alpha = fusion_aux.get("alpha")
        self.gamma = fusion_aux.get("gamma")
        self.beta = fusion_aux.get("beta")
        self.h_raw = fusion_aux.get("h_raw")
        self.h_context = fusion_aux.get("h_context")
        self.delta = fusion_aux.get("delta")
        self.attention_summary = fusion_aux.get("attention_summary")
        self.pooling = fusion_aux.get("pooling")
        self.num_latents = fusion_aux.get("num_latents")
        self.num_heads = fusion_aux.get("num_heads")
        self.num_layers = fusion_aux.get("num_layers")
        if self.alpha is not None:
            self.alpha_stats = self._summary_stats(self.alpha)
        if self.gamma is not None and self.beta is not None:
            gamma_stats = self._summary_stats(self.gamma)
            beta_stats = self._summary_stats(self.beta)
            self.gamma_beta_summary = {
                "gamma_l2_norm": gamma_stats["l2_norm"],
                "gamma_mean": gamma_stats["mean"],
                "beta_l2_norm": beta_stats["l2_norm"],
                "beta_mean": beta_stats["mean"],
            }

        extra_fusion_aux = {
            key: value
            for key, value in fusion_aux.items()
            if key not in known_fusion_keys
        }
        self.extra = extra_fusion_aux or None

    def populate_profiling(
        self,
        include_num_parameters: bool,
        include_embeddings: bool,
        num_parameters: dict[str, Any] | None = None,
        embeddings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Populate profiling diagnostics in-place."""
        if include_num_parameters:
            self.num_parameters = num_parameters
        if include_embeddings:
            self.embeddings = embeddings

    @classmethod
    def from_components(
        cls,
        *,
        logits: torch.Tensor,
        h_final: torch.Tensor,
        active_modalities: list[str],
        embedding_dim: int,
        fusion_aux: Mapping[str, Any],
        include_fusion_aux: bool,
        include_num_parameters: bool,
        include_embeddings: bool,
        num_parameters: Optional[dict[str, Any]] = None,
        embeddings: Optional[dict[str, torch.Tensor]] = None,
    ) -> "FusionAuxOutput":
        """Build a complete auxiliary output from model diagnostics."""

        output = cls(
            logits=logits,
            h_final=h_final,
            active_modalities=active_modalities,
            embedding_dim=embedding_dim,
        )
        output.populate_fusion(
            fusion_aux=fusion_aux,
            include_fusion_aux=include_fusion_aux,
        )
        output.populate_profiling(
            include_num_parameters=include_num_parameters,
            include_embeddings=include_embeddings,
            num_parameters=num_parameters,
            embeddings=embeddings,
        )
        return output
