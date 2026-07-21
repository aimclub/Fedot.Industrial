"""Tools for FUTURE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from typing import Any

import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@dataclass(frozen=True)
class AuxOutputConfig:
    """Configuration for auxiliary diagnostics payload."""

    include_embeddings: bool = True
    include_num_parameters: bool = True
    include_fusion_aux: bool = True


KNOWN_FUSION_KEYS = frozenset(
    {"gates", "alpha", "gamma", "beta", "h_raw", "h_context", "delta", "h_final"}
)


@dataclass
class FusionAuxOutput:
    """Typed auxiliary output for multimodal fusion classifier."""

    logits: torch.Tensor
    h_final: torch.Tensor
    active_modalities: list[str]
    embedding_dim: int
    num_parameters: dict[str, Any] | None = None
    embeddings: dict[str, torch.Tensor] | None = None
    gates: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    gamma: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    h_raw: torch.Tensor | None = None
    h_context: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    alpha_stats: dict[str, float] | None = None
    gamma_beta_summary: dict[str, float] | None = None
    extra: dict[str, Any] | None = None

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
