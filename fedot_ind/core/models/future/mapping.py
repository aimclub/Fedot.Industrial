"""Registries for FUTURE multimodal model composition."""

from dataclasses import dataclass

from fedot_ind.core.models.future.enums import FusionInputsParam
from fedot_ind.core.models.future.enums import FusionMethod
from fedot_ind.core.models.future.fusion.base import BaseFusionStrategy
from fedot_ind.core.models.future.fusion.bottleneck_fusion import (
    ContextOnlyResidualBottleneckFusion,
    OrdinaryBottleneckFusion,
    RawResidualBottleneckFusion,
)
from fedot_ind.core.models.future.fusion.concat_fusion import MultiConcatFusionMLP
from fedot_ind.core.models.future.fusion.film_fusion import FiLMFusion
from fedot_ind.core.models.future.fusion.gated_fusion import MultiModalGatedFusion
from fedot_ind.core.models.future.fusion.raw_centered_residual_fusion import (
    RawCenteredResidualFusion,
)


@dataclass(frozen=True)
class FusionRegistryEntry:
    """Registry descriptor for one FUTURE fusion strategy."""

    fusion_class: type[BaseFusionStrategy]
    inputs_param: FusionInputsParam
    requires_raw: bool = False
    is_extended: bool = False

    def resolve_input_count(self, n_modalities: int) -> int:
        if self.inputs_param.excludes_raw:
            return n_modalities - 1
        return n_modalities


FUSION_REGISTRY: dict[FusionMethod, FusionRegistryEntry] = {
    FusionMethod.concat: FusionRegistryEntry(
        MultiConcatFusionMLP,
        FusionInputsParam.n_inputs,
    ),
    FusionMethod.gated: FusionRegistryEntry(
        MultiModalGatedFusion,
        FusionInputsParam.n_inputs,
    ),
    FusionMethod.film: FusionRegistryEntry(
        FiLMFusion,
        FusionInputsParam.n_context_inputs,
        requires_raw=True,
    ),
    FusionMethod.raw_centered_residual: FusionRegistryEntry(
        RawCenteredResidualFusion,
        FusionInputsParam.n_context_inputs,
        requires_raw=True,
    ),
    FusionMethod.ordinary_bottleneck: FusionRegistryEntry(
        OrdinaryBottleneckFusion,
        FusionInputsParam.n_inputs,
        is_extended=True,
    ),
    FusionMethod.raw_residual_bottleneck: FusionRegistryEntry(
        RawResidualBottleneckFusion,
        FusionInputsParam.n_inputs,
        requires_raw=True,
        is_extended=True,
    ),
    FusionMethod.context_only_residual_bottleneck: FusionRegistryEntry(
        ContextOnlyResidualBottleneckFusion,
        FusionInputsParam.n_context_inputs,
        requires_raw=True,
        is_extended=True,
    ),
}
