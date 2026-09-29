"""Effect boundary that projects FEDOT and GOLEM graphs into pure specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fedot_ind.core.optimizer.graph_validation.capabilities import industrial_capability_index
from fedot_ind.core.optimizer.graph_validation.contracts import (
    ComputeDevice,
    GraphNodeSpec,
    GraphSpec,
    NodePosition,
    OperationCapabilities,
    OperationKind,
)


def graph_spec_from_runtime(
        graph: object,
        *,
        adapter: object | None = None,
        task_type: object | None = None,
        requested_device: object | None = None,
        require_serializable: bool = False,
        capability_index: Mapping[str, OperationCapabilities] | None = None,
) -> GraphSpec:
    """Restore a runtime graph once and extract only immutable validation data."""
    _, spec = project_runtime_graph(
        graph,
        adapter=adapter,
        task_type=task_type,
        requested_device=requested_device,
        require_serializable=require_serializable,
        capability_index=capability_index,
    )
    return spec


def project_runtime_graph(
        graph: object,
        *,
        adapter: object | None = None,
        task_type: object | None = None,
        requested_device: object | None = None,
        require_serializable: bool = False,
        capability_index: Mapping[str, OperationCapabilities] | None = None,
) -> tuple[object, GraphSpec]:
    """Return one restored runtime graph and its immutable validation projection."""
    pipeline = _restore_graph(graph, adapter)
    runtime_nodes = tuple(getattr(pipeline, "nodes", ()) or ())
    identities = {
        id(node): _runtime_node_id(node, index)
        for index, node in enumerate(runtime_nodes)
    }
    capabilities = capability_index if capability_index is not None else industrial_capability_index()
    nodes = tuple(
        _node_spec(node, identities, capabilities)
        for node in runtime_nodes
    )
    spec = GraphSpec(
        graph_id=_graph_id(graph, pipeline),
        nodes=nodes,
        task_type=_enum_name(task_type),
        requested_device=_device(requested_device),
        require_serializable=require_serializable,
    )
    return pipeline, spec


def infer_validation_context(graph_generation_params: object) -> tuple[str | None, str | None]:
    """Read task and device from explicit runtime collaborators without defaults."""
    advisor = getattr(graph_generation_params, "advisor", None)
    task = getattr(advisor, "task", None)
    task_type = getattr(task, "task_type", task)
    execution = getattr(graph_generation_params, "execution_policy", None)
    requested_device = getattr(execution, "device", None)
    return _enum_name(task_type), _enum_name(requested_device)


def capabilities_from_runtime_node(
        node: object,
        *,
        capability_index: Mapping[str, OperationCapabilities] | None = None,
) -> OperationCapabilities | None:
    """Resolve catalog capabilities first, then project FEDOT operation metadata."""
    operation = getattr(node, "operation", None)
    operation_name = _operation_name(node)
    declared = (capability_index or {}).get(operation_name)
    if declared is not None:
        return declared
    metadata = getattr(operation, "metadata", None)
    if metadata is None:
        content = getattr(node, "content", None)
        if isinstance(content, Mapping):
            metadata = content.get("metadata")
    if metadata is None:
        return None

    kind = _operation_kind(operation, operation_name)
    backend = _metadata_value(metadata, "backend", None)
    tags = _string_values(_metadata_value(metadata, "tags", ()))
    devices = ((ComputeDevice.CPU, ComputeDevice.CUDA)
               if backend == "torch" or "torch" in tags
               else (ComputeDevice.CPU,))
    return OperationCapabilities(
        name=operation_name,
        kind=kind,
        input_data_types=_enum_names(_metadata_value(metadata, "input_types", ())),
        output_data_types=_enum_names(_metadata_value(metadata, "output_types", ())),
        task_types=_enum_names(_metadata_value(metadata, "task_type", ())),
        devices=devices,
        serializable=True,
        allowed_positions=_positions(_metadata_value(metadata, "allowed_positions", ("any",))),
        tags=tags,
    )


def _restore_graph(graph: object, adapter: object | None) -> object:
    nodes = getattr(graph, "nodes", None)
    if nodes is not None and all(hasattr(node, "operation") for node in nodes):
        return graph
    if adapter is None:
        return graph
    restore = getattr(adapter, "restore", None)
    if not callable(restore):
        raise TypeError("Graph adapter must provide a callable restore method")
    return restore(graph)


def _node_spec(
        node: object,
        identities: Mapping[int, str],
        capabilities: Mapping[str, OperationCapabilities],
) -> GraphNodeSpec:
    parents = tuple(getattr(node, "nodes_from", ()) or ())
    parent_ids = tuple(
        identities[id(parent)] if id(parent) in identities else _runtime_node_id(parent, index)
        for index, parent in enumerate(parents)
    )
    return GraphNodeSpec(
        node_id=identities[id(node)],
        operation_name=_operation_name(node),
        parent_ids=parent_ids,
        capabilities=capabilities_from_runtime_node(
            node,
            capability_index=capabilities,
        ),
    )


def _runtime_node_id(node: object, index: int) -> str:
    uid = getattr(node, "uid", None)
    if uid is not None:
        return str(uid)
    descriptive_id = getattr(node, "descriptive_id", None)
    if descriptive_id:
        return str(descriptive_id)
    return f"node-{index}:{_operation_name(node)}"


def _operation_name(node: object) -> str:
    operation = getattr(node, "operation", None)
    operation_type = getattr(operation, "operation_type", None)
    if operation_type:
        return str(operation_type).split("/", maxsplit=1)[0]
    content = getattr(node, "content", None)
    if isinstance(content, Mapping) and content.get("name"):
        return str(content["name"]).split("/", maxsplit=1)[0]
    name = getattr(node, "name", None)
    return str(name or operation or type(node).__name__).split("/", maxsplit=1)[0]


def _operation_kind(operation: object, operation_name: str) -> OperationKind:
    if operation_name.startswith("data_source"):
        return OperationKind.DATA_SOURCE
    try:
        from fedot.core.operations.model import Model

        if isinstance(operation, Model):
            return OperationKind.MODEL
    except ImportError:
        pass
    return OperationKind.TRANSFORM


def _metadata_value(metadata: object, name: str, default: Any) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _enum_name(value: object | None) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return str(name)
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower()
    aliases = {
        "gpu": "cuda",
        "time_series": "ts",
        "table": "tabular",
    }
    return aliases.get(normalized, normalized) or None


def _enum_names(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        values = (values,)
    return tuple(dict.fromkeys(value for item in values
                               if (value := _enum_name(item)) is not None))


def _string_values(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        values = (values,)
    return tuple(dict.fromkeys(str(value) for value in values))


def _positions(values: object) -> tuple[NodePosition, ...]:
    aliases = {
        "any": NodePosition.ANY,
        "primary": NodePosition.PRIMARY,
        "secondary": NodePosition.SECONDARY,
        "root": NodePosition.ROOT,
    }
    if values is None:
        values = ()
    source = tuple(values) if not isinstance(values, (str, bytes)) and hasattr(values, "__iter__") else (values,)
    unknown = tuple(sorted({str(value).strip().lower() for value in source} - set(aliases)))
    if unknown:
        raise ValueError(f"Unsupported operation positions: {', '.join(unknown)}")
    positions = tuple(dict.fromkeys(
        aliases[str(value).strip().lower()]
        for value in source
        if str(value).strip().lower() in aliases
    ))
    return positions or (NodePosition.ANY,)


def _device(value: object | None) -> ComputeDevice | None:
    normalized = _enum_name(value)
    if normalized is None or normalized in ("auto", "default"):
        return None
    try:
        return ComputeDevice(normalized)
    except ValueError as error:
        raise ValueError(f"Unsupported validation device: {normalized}") from error


def _graph_id(graph: object, pipeline: object) -> str:
    for candidate in (graph, pipeline):
        graph_id = getattr(candidate, "uid", None)
        if graph_id:
            return str(graph_id)
    for candidate in (graph, pipeline):
        try:
            graph_id = getattr(candidate, "descriptive_id", None)
        except (AttributeError, RecursionError, TypeError, ValueError):
            graph_id = None
        if graph_id:
            return str(graph_id)
    return f"graph:{len(tuple(getattr(pipeline, 'nodes', ()) or ()))}"
