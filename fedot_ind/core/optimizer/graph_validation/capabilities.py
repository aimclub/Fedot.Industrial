"""Pure conversion of Industrial declarations into graph capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from fedot_ind.core.optimizer.graph_validation.contracts import (
    ComputeDevice,
    NodePosition,
    OperationCapabilities,
    OperationKind,
    StructuralRole,
)
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialOperationDeclaration,
    IndustrialOperationKind,
)


def capabilities_from_declaration(
        declaration: IndustrialOperationDeclaration,
) -> OperationCapabilities:
    """Translate the extension catalog contract without importing implementations."""
    return OperationCapabilities(
        name=declaration.name,
        kind=(OperationKind.MODEL
              if declaration.kind is IndustrialOperationKind.MODEL
              else OperationKind.TRANSFORM),
        input_data_types=declaration.data_types,
        output_data_types=(declaration.output_data_type,),
        task_types=declaration.tasks,
        devices=tuple(ComputeDevice(value) for value in declaration.devices),
        serializable=declaration.serializable,
        allowed_positions=tuple(NodePosition(value) for value in declaration.allowed_positions),
        min_parents=declaration.min_parents,
        max_parents=declaration.max_parents,
        allows_identical_parents=declaration.allows_identical_parents,
        supports_multimodal=declaration.supports_multimodal,
        requires_target=declaration.requires_target,
        requires_fit=declaration.requires_fit,
        tags=declaration.tags,
        structural_role=(None if declaration.structural_role is None
                         else StructuralRole(declaration.structural_role)),
    )


@lru_cache(maxsize=1)
def industrial_capability_index() -> Mapping[str, OperationCapabilities]:
    """Build a deterministic capability index from the validated data catalog."""
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog

    return MappingProxyType({
        declaration.name: capabilities_from_declaration(declaration)
        for declaration in load_industrial_extension_catalog().operations
    })
