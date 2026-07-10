"""Registries for FUTURE multimodal model composition."""

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.fusion.concat_fusion import MultiConcatFusionMLP
from fedot_ind.core.models.future.fusion.film_fusion import FiLMFusion
from fedot_ind.core.models.future.fusion.gated_fusion import MultiModalGatedFusion
from fedot_ind.core.models.future.fusion.raw_centered_residual_fusion import (
    RawCenteredResidualFusion,
)

FUSION_REGISTRY: dict[str, tuple[type[BaseFusionStrategy], str]] = {
    "concat": (MultiConcatFusionMLP, "n_inputs"),
    "gated": (MultiModalGatedFusion, "n_inputs"),
    "film": (FiLMFusion, "n_context_inputs"),
    "raw_centered_residual": (RawCenteredResidualFusion, "n_context_inputs"),
}

RAW_CENTERED_FUSION_METHODS = frozenset({"film", "raw_centered_residual"})
