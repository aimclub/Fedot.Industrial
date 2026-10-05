"""Pure capability-aware validation rules for evolutionary graph candidates."""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable, Iterable, Mapping

from fedot_ind.core.optimizer.graph_validation.contracts import (
    GraphNodeSpec,
    GraphSpec,
    NodePosition,
    OperationKind,
    StructuralRole,
    ValidationIssue,
    ValidationIssueCode,
    ValidationReport,
)


GraphRule = Callable[[GraphSpec], tuple[ValidationIssue, ...]]


def validate_graph(spec: GraphSpec, *, rules: Iterable[GraphRule] | None = None) -> ValidationReport:
    """Return sorted issues from the selected graph rules.

    ``rules=None`` selects the default pure rules; an explicit iterable
    replaces them, and an empty iterable produces a valid report. Exceptions
    from rules propagate.
    """
    selected_rules = tuple(rules) if rules is not None else DEFAULT_GRAPH_RULES
    issues = tuple(sorted(
        (issue for rule in selected_rules for issue in rule(spec)),
        key=_issue_sort_key,
    ))
    return ValidationReport(graph_id=spec.graph_id, issues=issues)


def validate_structure(spec: GraphSpec) -> tuple[ValidationIssue, ...]:
    """Return issues for empty, cyclic, disconnected, or ambiguously rooted graphs.

    Also report duplicate node IDs, absent parents, self-links, isolated nodes,
    and missing primary nodes. A root is a node with no children.
    """
    if not spec.nodes:
        return (_issue(ValidationIssueCode.EMPTY_GRAPH, "Graph has no nodes."),)

    issues: list[ValidationIssue] = []
    counts = Counter(spec.node_ids)
    duplicate_ids = tuple(sorted(node_id for node_id, count in counts.items() if count > 1))
    if duplicate_ids:
        issues.append(_issue(
            ValidationIssueCode.DUPLICATE_NODE_ID,
            "Graph contains duplicate node identifiers.",
            node_ids=duplicate_ids,
        ))

    known_ids = frozenset(spec.node_ids)
    for node in spec.nodes:
        unknown = tuple(sorted(parent_id for parent_id in node.parent_ids if parent_id not in known_ids))
        if unknown:
            issues.append(_issue(
                ValidationIssueCode.UNKNOWN_PARENT,
                "Node references parents that are absent from the graph.",
                node_ids=(node.node_id, *unknown),
                operation_names=(node.operation_name,),
            ))
        if node.node_id in node.parent_ids:
            issues.append(_issue(
                ValidationIssueCode.SELF_CYCLE,
                "Node references itself as a parent.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
            ))

    unique_nodes = _unique_node_map(spec)
    cycle_nodes = _cycle_nodes(unique_nodes)
    if cycle_nodes:
        issues.append(_issue(
            ValidationIssueCode.CYCLE,
            "Graph contains a directed cycle.",
            node_ids=cycle_nodes,
        ))

    roots = _root_ids(unique_nodes)
    if len(roots) != 1:
        issues.append(_issue(
            ValidationIssueCode.ROOT_COUNT,
            "Graph must contain exactly one root node.",
            node_ids=roots,
            context={"root_count": len(roots)},
        ))

    if len(unique_nodes) > 1:
        neighbours = _undirected_neighbours(unique_nodes)
        isolated = tuple(sorted(node_id for node_id, values in neighbours.items() if not values))
        if isolated:
            issues.append(_issue(
                ValidationIssueCode.ISOLATED_NODE,
                "Graph contains isolated nodes.",
                node_ids=isolated,
            ))
        components = _components(neighbours)
        if len(components) > 1:
            issues.append(_issue(
                ValidationIssueCode.DISCONNECTED_COMPONENT,
                "Graph contains disconnected components.",
                node_ids=tuple(node_id for component in components for node_id in component),
                context={"component_count": len(components)},
            ))

    if unique_nodes and not any(not node.parent_ids for node in unique_nodes.values()):
        issues.append(_issue(
            ValidationIssueCode.MISSING_PRIMARY_NODE,
            "Graph has no primary node without parents.",
        ))
    return tuple(issues)


def validate_capabilities(spec: GraphSpec) -> tuple[ValidationIssue, ...]:
    """Return operation capability violations for the graph's execution context.

    Check supported positions, parent counts, tasks, devices, serialization,
    and model roots. Missing capability declarations are reported as issues.
    """
    issues: list[ValidationIssue] = []
    nodes = _unique_node_map(spec)
    roots = frozenset(_root_ids(nodes))
    for node in nodes.values():
        capability = node.capabilities
        if capability is None:
            issues.append(_issue(
                ValidationIssueCode.UNKNOWN_OPERATION,
                "Operation capabilities are unavailable.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
            ))
            continue

        parent_count = len(node.parent_ids)
        positions = _node_positions(node, roots)
        if not capability.supports_any_position(positions):
            issues.append(_issue(
                ValidationIssueCode.POSITION_NOT_ALLOWED,
                "Operation is used in a graph position it does not support.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
                context={"positions": [position.value for position in positions]},
            ))
        if parent_count < capability.min_parents:
            issues.append(_issue(
                ValidationIssueCode.TOO_FEW_PARENTS,
                "Operation has fewer parents than its capability contract requires.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
                context={"actual": parent_count, "minimum": capability.min_parents},
            ))
        if capability.max_parents is not None and parent_count > capability.max_parents:
            issues.append(_issue(
                ValidationIssueCode.TOO_MANY_PARENTS,
                "Operation has more parents than its capability contract allows.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
                context={"actual": parent_count, "maximum": capability.max_parents},
            ))
        if spec.task_type and capability.task_types and spec.task_type not in capability.task_types:
            if not _is_class_decompose_regression_branch(node, nodes, spec.task_type):
                issues.append(_issue(
                    ValidationIssueCode.TASK_NOT_SUPPORTED,
                    "Operation does not support the graph task.",
                    node_ids=(node.node_id,),
                    operation_names=(node.operation_name,),
                    context={"task_type": spec.task_type,
                             "supported": list(capability.task_types)},
                ))
        if spec.requested_device is not None and spec.requested_device not in capability.devices:
            issues.append(_issue(
                ValidationIssueCode.DEVICE_NOT_SUPPORTED,
                "Operation does not support the requested execution device.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
                context={"device": spec.requested_device.value,
                         "supported": [device.value for device in capability.devices]},
            ))
        if spec.require_serializable and not capability.serializable:
            issues.append(_issue(
                ValidationIssueCode.OPERATION_NOT_SERIALIZABLE,
                "Operation cannot be serialized under the requested graph contract.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
            ))

    for root_id in sorted(roots):
        root = nodes[root_id]
        if root.capabilities is not None and root.capabilities.kind is not OperationKind.MODEL:
            issues.append(_issue(
                ValidationIssueCode.ROOT_NOT_MODEL,
                "The final graph operation must be a model.",
                node_ids=(root.node_id,),
                operation_names=(root.operation_name,),
            ))
    return tuple(issues)


def validate_data_flow(spec: GraphSpec) -> tuple[ValidationIssue, ...]:
    """Return incompatible parent-output/child-input and duplicate-parent issues.

    Skip type comparisons when either endpoint lacks capabilities or either
    type declaration is empty.
    """
    issues: list[ValidationIssue] = []
    nodes = _unique_node_map(spec)
    for child in nodes.values():
        child_capability = child.capabilities
        if child_capability is None:
            continue
        parent_names: list[str] = []
        for parent_id in child.parent_ids:
            parent = nodes.get(parent_id)
            if parent is None or parent.capabilities is None:
                continue
            parent_names.append(parent.operation_name)
            output_types = frozenset(parent.capabilities.output_data_types)
            input_types = frozenset(child_capability.input_data_types)
            if output_types and input_types and output_types.isdisjoint(input_types):
                issues.append(_issue(
                    ValidationIssueCode.DATA_TYPE_MISMATCH,
                    "Parent output types are incompatible with child input types.",
                    node_ids=(parent.node_id, child.node_id),
                    operation_names=(parent.operation_name, child.operation_name),
                    context={"parent_outputs": sorted(output_types),
                             "child_inputs": sorted(input_types)},
                ))
        parent_name_counts = Counter(parent_names)
        repeated_transform_names = {
            parent.operation_name
            for parent_id in child.parent_ids
            if (parent := nodes.get(parent_id)) is not None
            and parent.capabilities is not None
            and parent.capabilities.kind is OperationKind.TRANSFORM
            and parent_name_counts[parent.operation_name] > 1
        }
        if (len(parent_names) > 1
                and len(set(parent_names)) != len(parent_names)
                and (not child_capability.allows_identical_parents or repeated_transform_names)):
            issues.append(_issue(
                ValidationIssueCode.IDENTICAL_PARENT_OPERATIONS,
                "Operation cannot combine identical parent operations.",
                node_ids=(child.node_id,),
                operation_names=(child.operation_name, *parent_names),
            ))
    return tuple(issues)


def validate_special_operations(spec: GraphSpec) -> tuple[ValidationIssue, ...]:
    """Return violations of source, resampling, decomposition, and filtering rules."""
    issues: list[ValidationIssue] = []
    nodes = _unique_node_map(spec)
    children = _children_by_parent(nodes)
    primary_nodes = tuple(node for node in nodes.values() if not node.parent_ids)

    source_flags = tuple(node.capabilities is not None
                         and node.capabilities.kind is OperationKind.DATA_SOURCE
                         for node in primary_nodes)
    if source_flags and any(source_flags) and not all(source_flags):
        issues.append(_issue(
            ValidationIssueCode.MIXED_DATA_SOURCES,
            "Data-source and ordinary primary nodes cannot be mixed.",
            node_ids=(node.node_id for node in primary_nodes),
            operation_names=(node.operation_name for node in primary_nodes),
        ))

    resample_nodes = tuple(node for node in nodes.values()
                           if node.capabilities is not None
                           and node.capabilities.structural_role is StructuralRole.RESAMPLING)
    for node in resample_nodes:
        if node.parent_ids or (len(primary_nodes) > 1):
            issues.append(_issue(
                ValidationIssueCode.RESAMPLE_POSITION,
                "Resample must be the only primary operation.",
                node_ids=(node.node_id,),
                operation_names=(node.operation_name,),
            ))
        for child_id in children.get(node.node_id, ()):
            child = nodes[child_id]
            if len(child.parent_ids) > 1:
                issues.append(_issue(
                    ValidationIssueCode.RESAMPLE_PARENT_CONFLICT,
                    "Resample must be the single parent of its child.",
                    node_ids=(node.node_id, child.node_id),
                    operation_names=(node.operation_name, child.operation_name),
                ))

    for node in nodes.values():
        if (node.capabilities is None or node.capabilities.structural_role not in
                (StructuralRole.DECOMPOSITION, StructuralRole.CLASS_DECOMPOSITION)):
            continue
        if len(node.parent_ids) != 2:
            issues.append(_issue(
                ValidationIssueCode.DECOMPOSE_PARENT_COUNT,
                "A decomposition operation requires exactly two parents.",
                node_ids=(node.node_id, *node.parent_ids),
                operation_names=(node.operation_name,),
                context={"actual": len(node.parent_ids)},
            ))
            continue
        model_parent = nodes.get(node.parent_ids[0])
        if (model_parent is None or model_parent.capabilities is None
                or model_parent.capabilities.kind is not OperationKind.MODEL):
            issues.append(_issue(
                ValidationIssueCode.DECOMPOSE_MODEL_PARENT_REQUIRED,
                "The first decomposition parent must be a model.",
                node_ids=(node.node_id, node.parent_ids[0]),
                operation_names=(node.operation_name,),
            ))

    conflicted = _parallel_filtering_nodes(nodes)
    if conflicted:
        issues.append(_issue(
            ValidationIssueCode.PARALLEL_FILTERING_BRANCHES,
            "Parallel branches containing filtering operations cannot be merged.",
            node_ids=conflicted,
        ))
    return tuple(issues)


DEFAULT_GRAPH_RULES: tuple[GraphRule, ...] = (
    validate_structure,
    validate_capabilities,
    validate_data_flow,
    validate_special_operations,
)


def _issue(code: ValidationIssueCode, message: str, **values) -> ValidationIssue:
    return ValidationIssue.create(code, message, **values)


def _issue_sort_key(issue: ValidationIssue) -> tuple[object, ...]:
    return (
        issue.severity.value,
        issue.code.value,
        issue.node_ids,
        issue.operation_names,
        repr(issue.context),
    )


def _unique_node_map(spec: GraphSpec) -> dict[str, GraphNodeSpec]:
    nodes: dict[str, GraphNodeSpec] = {}
    for node in spec.nodes:
        nodes.setdefault(node.node_id, node)
    return nodes


def _children_by_parent(nodes: Mapping[str, GraphNodeSpec]) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in nodes.values():
        for parent_id in node.parent_ids:
            if parent_id in children:
                children[parent_id].append(node.node_id)
    return {node_id: tuple(sorted(values)) for node_id, values in children.items()}


def _root_ids(nodes: Mapping[str, GraphNodeSpec]) -> tuple[str, ...]:
    parent_ids = {parent_id for node in nodes.values() for parent_id in node.parent_ids}
    return tuple(sorted(set(nodes) - parent_ids))


def _cycle_nodes(nodes: Mapping[str, GraphNodeSpec]) -> tuple[str, ...]:
    indegree = {node_id: 0 for node_id in nodes}
    children = _children_by_parent(nodes)
    for node in nodes.values():
        indegree[node.node_id] = sum(parent_id in nodes for parent_id in node.parent_ids)
    ready = deque(sorted(node_id for node_id, count in indegree.items() if count == 0))
    visited: set[str] = set()
    while ready:
        node_id = ready.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        for child_id in children[node_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    return tuple(sorted(set(nodes) - visited))


def _undirected_neighbours(nodes: Mapping[str, GraphNodeSpec]) -> dict[str, set[str]]:
    neighbours = {node_id: set() for node_id in nodes}
    for node in nodes.values():
        for parent_id in node.parent_ids:
            if parent_id in nodes and parent_id != node.node_id:
                neighbours[node.node_id].add(parent_id)
                neighbours[parent_id].add(node.node_id)
    return neighbours


def _components(neighbours: Mapping[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    remaining = set(neighbours)
    components: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        pending = [start]
        component: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in component:
                continue
            component.add(node_id)
            pending.extend(sorted(neighbours[node_id] - component, reverse=True))
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _node_positions(node: GraphNodeSpec, roots: frozenset[str]) -> tuple[NodePosition, ...]:
    positions = []
    if node.node_id in roots:
        positions.append(NodePosition.ROOT)
    positions.append(NodePosition.PRIMARY if not node.parent_ids else NodePosition.SECONDARY)
    return tuple(positions)


def _is_class_decompose_regression_branch(
        node: GraphNodeSpec,
        nodes: Mapping[str, GraphNodeSpec],
        task_type: str,
) -> bool:
    if task_type != "classification" or node.capabilities is None:
        return False
    if "regression" not in node.capabilities.task_types:
        return False
    return any(nodes.get(parent_id) is not None
               and nodes[parent_id].capabilities is not None
               and nodes[parent_id].capabilities.structural_role is StructuralRole.CLASS_DECOMPOSITION
               for parent_id in node.parent_ids)


def _parallel_filtering_nodes(nodes: Mapping[str, GraphNodeSpec]) -> tuple[str, ...]:
    contains_filtering: dict[str, bool] = {}
    visiting: set[str] = set()
    conflicts: set[str] = set()

    def inspect(node_id: str) -> bool:
        if node_id in contains_filtering:
            return contains_filtering[node_id]
        if node_id in visiting:
            return False
        visiting.add(node_id)
        node = nodes[node_id]
        parent_flags = tuple(inspect(parent_id) for parent_id in node.parent_ids if parent_id in nodes)
        if sum(parent_flags) > 1:
            conflicts.add(node_id)
        own = bool(node.capabilities is not None and "filtering" in node.capabilities.tags)
        contains_filtering[node_id] = own or any(parent_flags)
        visiting.remove(node_id)
        return contains_filtering[node_id]

    for key in sorted(nodes):
        inspect(key)
    return tuple(sorted(conflicts))
