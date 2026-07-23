"""Registries for FUTURE multimodal model composition."""

from dataclasses import dataclass

from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.fusion.concat_fusion import MultiConcatFusionMLP
from fedot_ind.core.models.future.enums import FusionMethod
from fedot_ind.core.models.future.fusion.film_fusion import FiLMFusion
from fedot_ind.core.models.future.fusion.gated_fusion import MultiModalGatedFusion
from fedot_ind.core.models.future.fusion.raw_centered_residual_fusion import (
    RawCenteredResidualFusion,
)


@dataclass(frozen=True)
class FusionRegistryEntry:
    """Registry descriptor for one FUTURE fusion strategy."""

    fusion_class: type[BaseFusionStrategy]
    inputs_param_name: str
    requires_raw: bool = False

    def resolve_input_count(self, n_modalities: int) -> int:
        return n_modalities - 1 if self.requires_raw else n_modalities


FUSION_REGISTRY: dict[FusionMethod, FusionRegistryEntry] = {
    FusionMethod.concat: FusionRegistryEntry(MultiConcatFusionMLP, "n_inputs"),
    FusionMethod.gated: FusionRegistryEntry(MultiModalGatedFusion, "n_inputs"),
    FusionMethod.film: FusionRegistryEntry(
        FiLMFusion,
        "n_context_inputs",
        requires_raw=True,
    ),
    FusionMethod.raw_centered_residual: FusionRegistryEntry(
        RawCenteredResidualFusion,
        "n_context_inputs",
        requires_raw=True,
    ),
}
