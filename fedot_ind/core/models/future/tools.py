"""Tools for FUTURE models."""

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
    extra: dict[str, Any] | None = None

    def populate_fusion(
        self,
        fusion_aux: Mapping[str, Any],
        include_fusion_aux: bool,
        known_fusion_keys: set[str] | frozenset[str],
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