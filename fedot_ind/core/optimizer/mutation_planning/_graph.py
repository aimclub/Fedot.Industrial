"""Local graph transitions; no runtime objects, catalog discovery or randomness."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from hashlib import sha256
import json

from fedot_ind.core.optimizer.graph_validation.contracts import (
    GraphNodeSpec, GraphSpec, OperationCapabilities, OperationKind,
)
from fedot_ind.core.optimizer.graph_validation.validation import validate_graph, validate_structure
from fedot_ind.core.optimizer.mutation_planning.contracts import (
    AddBranch, AddPrimaryBranch, DeduplicateParents, DropPrimaryBranch, DropPrimaryTransform, DropTransform, GraphEdit,
    InsertTransform, MutationLimits, MutationRejection, MutationRejectionCode,
    RawInputSpec, ReplaceOperation,
)


def graph_signature(spec: GraphSpec) -> str:
    """Include capabilities and context so a plan cannot silently change its base."""
    payload = json.dumps(spec.to_record(), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def candidate_signature(spec: GraphSpec) -> str:
    """Canonical ordered-root traversal, preserving sharing but ignoring edit IDs.

    Parent order is semantic. This needs no graph permutations or isomorphism
    search: a valid graph has one root and ordered incoming edges.
    """
    nodes = {node.node_id: node for node in spec.nodes}
    parents = {parent for node in spec.nodes for parent in node.parent_ids}
    root = next(node.node_id for node in spec.nodes if node.node_id not in parents)
    pending, order, identifiers = [root], [], {}
    while pending:
        node_id = pending.pop()
        if node_id not in identifiers:
            identifiers[node_id] = f"node-{len(order)}"
            order.append(node_id)
            pending.extend(reversed(nodes[node_id].parent_ids))
    canonical = replace(spec, graph_id="candidate", nodes=tuple(replace(
        nodes[node_id], node_id=identifiers[node_id],
        parent_ids=tuple(identifiers[parent] for parent in nodes[node_id].parent_ids))
        for node_id in order))
    return graph_signature(canonical)


def graph_depth(spec: GraphSpec) -> int | None:
    """Linear topological depth; None means ambiguous IDs, absent parents or a cycle."""
    nodes = {node.node_id: node for node in spec.nodes}
    if len(nodes) != len(spec.nodes):
        return None
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    pending = {}
    for node in spec.nodes:
        parents = tuple(dict.fromkeys(node.parent_ids))
        if any(parent not in nodes for parent in parents):
            return None
        pending[node.node_id] = len(parents)
        for parent in parents:
            children[parent].append(node.node_id)
    ready = deque(node_id for node_id in nodes if not pending[node_id])
    depths = {node_id: 1 for node_id in ready}
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for child in children[node_id]:
            depths[child] = max(depths.get(child, 1), depths[node_id] + 1)
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return max(depths.values(), default=0) if visited == len(nodes) else None


def budget_excess(spec: GraphSpec, limits: MutationLimits) -> tuple[int, int, int]:
    depth = graph_depth(spec)
    return (max(0, len(spec.nodes) - limits.max_nodes),
            max(0, (depth or 0) - limits.max_depth),
            sum(max(0, len(node.parent_ids) - limits.max_arity) for node in spec.nodes)
            if limits.max_arity is not None else 0)


def ordinary(capability: OperationCapabilities | None) -> bool:
    return bool(capability is not None
                and capability.kind in (OperationKind.MODEL, OperationKind.TRANSFORM)
                and capability.structural_role is None
                and "filtering" not in capability.tags
                and capability.input_data_types and capability.output_data_types)


def single_input_graph(spec: GraphSpec) -> bool:
    """A multimodal-capable operation is not itself a multimodal input context."""
    return (sum(not node.parent_ids for node in spec.nodes) == 1
            and all(len(node.parent_ids) <= 1 and node.capabilities is not None
                    and node.capabilities.kind is not OperationKind.DATA_SOURCE
                    and node.capabilities.structural_role is None for node in spec.nodes))


def homogeneous_input_graph(spec: GraphSpec, raw_input: RawInputSpec | None) -> bool:
    """Multiple primary transforms may still consume one homogeneous sample set."""
    return bool(raw_input is not None and spec.task_type in ("classification", "regression")
                and all(ordinary(node.capabilities)
                        and (node.parent_ids or raw_input.data_type in node.capabilities.input_data_types)
                        and (raw_input.has_target or not node.capabilities.requires_target)
                        for node in spec.nodes))


def unary_context_allowed(spec: GraphSpec, *capabilities: OperationCapabilities,
                          raw_input: RawInputSpec | None = None) -> bool:
    return (not any(capability.supports_multimodal for capability in capabilities)
            or single_input_graph(spec) or homogeneous_input_graph(spec, raw_input))


def graph_precondition(spec: GraphSpec, limits: MutationLimits,
                       *, check_budget: bool = True,
                       allow_duplicate_parents: bool = False,
                       raw_input: RawInputSpec | None = None) -> MutationRejection | None:
    # Reject structural ambiguity before special-operation rules traverse ancestors.
    structure = validate_graph(spec, rules=(validate_structure,))
    if not structure.is_valid:
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Structural ambiguity cannot be repaired locally.", structure)
    if spec.task_type is None:
        return MutationRejection(MutationRejectionCode.MISSING_CONTEXT,
                                 "An explicit graph task is required for structural edits.")
    if any(node.capabilities is None
           or node.capabilities.kind is OperationKind.UNKNOWN
           or node.capabilities.name != node.operation_name for node in spec.nodes):
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Unknown or inconsistent operation contracts are preserved, not inferred.")
    if not allow_duplicate_parents and any(
            len(set(node.parent_ids)) != len(node.parent_ids) for node in spec.nodes):
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Repeated parent references require explicit normalization.")
    if raw_input is not None:
        if not isinstance(raw_input, RawInputSpec):
            raise TypeError("raw_input must be a RawInputSpec or None")
        if any(node.capabilities.kind is OperationKind.DATA_SOURCE
               or node.capabilities.structural_role is not None for node in spec.nodes):
            return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                     "Homogeneous raw input cannot describe source, multimodal or special graphs.")
        if not unary_context_allowed(spec, *(node.capabilities for node in spec.nodes), raw_input=raw_input):
            return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                     "Multimodal-capable operations require a confirmed unary graph context.")
        if any((not node.parent_ids and raw_input.data_type not in node.capabilities.input_data_types)
               or (node.capabilities.requires_target and not raw_input.has_target) for node in spec.nodes):
            return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                     "The confirmed raw input does not satisfy the operation contracts.")
    if check_budget and any(budget_excess(spec, limits)):
        return MutationRejection(MutationRejectionCode.BUDGET_EXCEEDED,
                                 "Graph exceeds the node, depth or arity budget.")
    return None


def replacement_allowed(spec: GraphSpec, node: GraphNodeSpec,
                        operation: OperationCapabilities, raw_input: RawInputSpec | None = None) -> bool:
    """Type tuples declare supported alternatives, not guaranteed simultaneous types."""
    old = node.capabilities
    return bool(ordinary(old) and ordinary(operation)
                and unary_context_allowed(spec, old, operation, raw_input=raw_input)
                and operation.name != node.operation_name
                and operation.kind is old.kind
                and spec.task_type in old.task_types
                and spec.task_type in operation.task_types
                and not set(old.input_data_types).isdisjoint(operation.input_data_types)
                and not set(operation.output_data_types).isdisjoint(old.output_data_types)
                and not (operation.requires_target and not old.requires_target))


def drop_allowed(spec: GraphSpec, node: GraphNodeSpec, raw_input: RawInputSpec | None = None) -> bool:
    if (not ordinary(node.capabilities)
            or node.capabilities.kind is not OperationKind.TRANSFORM
            or len(node.parent_ids) != 1):
        return False
    nodes = {item.node_id: item for item in spec.nodes}
    parent = nodes.get(node.parent_ids[0])
    children = [item for item in spec.nodes if node.node_id in item.parent_ids]
    return bool(parent is not None and parent.capabilities is not None and children
                and spec.task_type in node.capabilities.task_types
                and all(ordinary(child.capabilities)
                        and unary_context_allowed(spec, node.capabilities, parent.capabilities,
                                                  child.capabilities, raw_input=raw_input)
                        and spec.task_type in child.capabilities.task_types
                        and parent.node_id not in child.parent_ids for child in children))


def drop_primary_allowed(spec: GraphSpec, node: GraphNodeSpec,
                         raw_input: RawInputSpec | None) -> bool:
    if (raw_input is None or node.parent_ids or not ordinary(node.capabilities)
            or node.capabilities.kind is not OperationKind.TRANSFORM
            or raw_input.data_type not in node.capabilities.input_data_types):
        return False
    if any(item.capabilities is None or item.capabilities.kind is OperationKind.DATA_SOURCE
           or item.capabilities.structural_role is not None
           for item in spec.nodes):
        return False
    if not unary_context_allowed(spec, *(item.capabilities for item in spec.nodes), raw_input=raw_input):
        return False
    children = [child for child in spec.nodes if node.node_id in child.parent_ids]
    return bool(children and spec.task_type in node.capabilities.task_types
                and all(child.parent_ids == (node.node_id,) and ordinary(child.capabilities)
                        and spec.task_type in child.capabilities.task_types
                        and raw_input.data_type in child.capabilities.input_data_types
                        and (raw_input.has_target or not child.capabilities.requires_target)
                        for child in children))


def new_node_id(spec: GraphSpec, parent_id: str | None, child_id: str,
                operation_name: str) -> str:
    payload = json.dumps((parent_id, child_id, operation_name), separators=(",", ":"))
    prefix = "mutation-" + sha256(payload.encode("utf-8")).hexdigest()[:16]
    known = set(spec.node_ids)
    for index in range(len(known) + 1):
        candidate = f"{prefix}-{index}"
        if candidate not in known:
            return candidate
    raise AssertionError("Finite ID allocation exhausted")


def drop_primary_branch_allowed(spec: GraphSpec, node: GraphNodeSpec, raw_input: RawInputSpec | None) -> bool:
    if (not homogeneous_input_graph(spec, raw_input) or node.parent_ids
            or node.capabilities.kind is not OperationKind.TRANSFORM
            or node.capabilities.output_data_types != ("tabular",)):
        return False
    children = [child for child in spec.nodes if node.node_id in child.parent_ids]
    nodes = {item.node_id: item for item in spec.nodes}
    return bool(len(children) == 1 and len(children[0].parent_ids) == 2
                and all(nodes[parent_id].capabilities.kind is OperationKind.TRANSFORM
                        and nodes[parent_id].capabilities.output_data_types == ("tabular",)
                        for parent_id in children[0].parent_ids))


def apply_edit(spec: GraphSpec, edit: GraphEdit, *,
               raw_input: RawInputSpec | None = None) -> GraphSpec | MutationRejection:
    """Check semantic safety before constructing a new immutable graph."""
    nodes = {node.node_id: node for node in spec.nodes}
    node = nodes.get(edit.node_id)
    invalid = MutationRejection(MutationRejectionCode.UNSUPPORTED_EDIT,
                                "Edit is absent, ambiguous or outside the safe local rules.")
    if isinstance(edit, (InsertTransform, AddPrimaryBranch)) and edit.operation.kind is not OperationKind.TRANSFORM:
        return MutationRejection(MutationRejectionCode.OPERATION_KIND, "Only feature transforms may be inserted.")
    if isinstance(edit, (ReplaceOperation, InsertTransform, AddBranch, AddPrimaryBranch)):
        if not ordinary(edit.operation):
            return MutationRejection(MutationRejectionCode.PROTECTED_OPERATION,
                                     "Special, source, filtering or unknown operations are not inserted locally.")
        if spec.task_type not in edit.operation.task_types:
            return MutationRejection(MutationRejectionCode.TASK_NOT_SUPPORTED, "Operation does not support this task.")
    if isinstance(edit, ReplaceOperation):
        if node is not None and node.capabilities is not None:
            if edit.operation.name == node.operation_name:
                return MutationRejection(MutationRejectionCode.SAME_OPERATION,
                                         "Replacement must change the operation.")
            if edit.operation.kind is not node.capabilities.kind:
                return MutationRejection(MutationRejectionCode.OPERATION_KIND,
                                         "Replacement must preserve operation kind.")
            if not unary_context_allowed(spec, node.capabilities, edit.operation, raw_input=raw_input):
                return MutationRejection(MutationRejectionCode.INPUT_CONTEXT,
                                         "Operation input context is unconfirmed.")
            if (set(node.capabilities.input_data_types).isdisjoint(edit.operation.input_data_types)
                    or set(node.capabilities.output_data_types).isdisjoint(edit.operation.output_data_types)):
                return MutationRejection(MutationRejectionCode.DATA_TYPE,
                                         "Replacement input/output types do not overlap.")
        if node is None or not replacement_allowed(spec, node, edit.operation, raw_input):
            return invalid
        updated = replace(node, operation_name=edit.operation.name, capabilities=edit.operation)
        return replace(spec, nodes=tuple(updated if item is node else item for item in spec.nodes))
    if isinstance(edit, InsertTransform):
        parent, child = nodes.get(edit.parent_id), nodes.get(edit.child_id)
        operation = edit.operation
        if child is not None and child.capabilities is not None:
            if not unary_context_allowed(spec, child.capabilities, operation, raw_input=raw_input):
                return MutationRejection(MutationRejectionCode.INPUT_CONTEXT,
                                         "Operation input context is unconfirmed.")
            if set(operation.output_data_types).isdisjoint(child.capabilities.input_data_types):
                return MutationRejection(MutationRejectionCode.DATA_TYPE, "Inserted output cannot feed the child.")
            if operation.requires_target and not child.capabilities.requires_target:
                return MutationRejection(MutationRejectionCode.TARGET_REQUIRED,
                                         "Insertion introduces a target requirement.")
        if (parent is not None and parent.capabilities is not None
                and set(parent.capabilities.output_data_types).isdisjoint(operation.input_data_types)):
            return MutationRejection(MutationRejectionCode.DATA_TYPE, "Inserted operation cannot consume the parent.")
        if (node is not None or child is None or not ordinary(child.capabilities)
                or not ordinary(operation) or operation.kind is not OperationKind.TRANSFORM
                or not unary_context_allowed(spec, child.capabilities, operation, raw_input=raw_input)
                or spec.task_type not in operation.task_types
                or spec.task_type not in child.capabilities.task_types
                or (operation.requires_target and not child.capabilities.requires_target)
                or set(operation.output_data_types).isdisjoint(child.capabilities.input_data_types)):
            return invalid
        if edit.parent_id is None:
            if (child.parent_ids
                    or set(child.capabilities.input_data_types).isdisjoint(operation.input_data_types)):
                return invalid
            parent_ids, child_parents = (), (edit.node_id,)
        else:
            if (parent is None or child.parent_ids.count(edit.parent_id) != 1
                    or parent.capabilities is None
                    or parent.capabilities.kind is OperationKind.UNKNOWN
                    or parent.capabilities.structural_role is not None
                    or not unary_context_allowed(spec, parent.capabilities, raw_input=raw_input)
                    or not parent.capabilities.output_data_types
                    or set(parent.capabilities.output_data_types).isdisjoint(operation.input_data_types)):
                return invalid
            parent_ids = (edit.parent_id,)
            child_parents = tuple(edit.node_id if parent_id == edit.parent_id else parent_id
                                  for parent_id in child.parent_ids)
        inserted = GraphNodeSpec(edit.node_id, operation.name, parent_ids, operation)
        updated = replace(child, parent_ids=child_parents)
        result = tuple(updated if item is child else item for item in spec.nodes)
        return replace(spec, nodes=result + (inserted,))
    if isinstance(edit, DropTransform):
        if node is None or not drop_allowed(spec, node, raw_input):
            return invalid
        return replace(spec, nodes=tuple(replace(item, parent_ids=tuple(
            node.parent_ids[0] if parent_id == node.node_id else parent_id
            for parent_id in item.parent_ids)) for item in spec.nodes if item is not node))
    if isinstance(edit, AddBranch):
        parent, child, operation = nodes.get(edit.parent_id), nodes.get(edit.child_id), edit.operation
        multimodal = any(item is not None and item.supports_multimodal for item in (
            operation, parent.capabilities if parent else None, child.capabilities if child else None))
        confirmed_features = (homogeneous_input_graph(spec, raw_input)
                              and operation.kind is OperationKind.TRANSFORM
                              and operation.output_data_types == ("tabular",)
                              and parent is not None and parent.capabilities is not None
                              and parent.capabilities.kind is OperationKind.TRANSFORM
                              and parent.capabilities.output_data_types == ("tabular",))
        if multimodal and not confirmed_features:
            return MutationRejection(MutationRejectionCode.INPUT_CONTEXT,
                                     "Branching multimodal-capable operations needs a homogeneous feature context.")
        if (node is not None or parent is None or child is None
                or child.parent_ids.count(edit.parent_id) != 1
                or not ordinary(child.capabilities) or not ordinary(operation)
                or parent.capabilities is None or parent.capabilities.kind is OperationKind.UNKNOWN
                or parent.capabilities.structural_role is not None
                or not parent.capabilities.output_data_types
                or spec.task_type not in operation.task_types
                or spec.task_type not in child.capabilities.task_types
                or (operation.requires_target and not child.capabilities.requires_target)
                or set(parent.capabilities.output_data_types).isdisjoint(operation.input_data_types)):
            return invalid
        # FEDOT requires one common type at an ordinary merge, not just valid pairs.
        common_types = set(child.capabilities.input_data_types) & set(operation.output_data_types)
        for parent_id in child.parent_ids:
            previous = nodes.get(parent_id)
            if (previous is None or previous.capabilities is None
                    or previous.capabilities.structural_role is not None
                    or (previous.capabilities.supports_multimodal and not confirmed_features)):
                return invalid
            common_types.intersection_update(previous.capabilities.output_data_types)
        if not common_types:
            return MutationRejection(MutationRejectionCode.DATA_TYPE,
                                     "All feature branches need one common merge type.")
        branch = GraphNodeSpec(edit.node_id, operation.name, (edit.parent_id,), operation)
        updated = replace(child, parent_ids=child.parent_ids + (edit.node_id,))
        return replace(spec, nodes=tuple(updated if item is child else item
                                         for item in spec.nodes) + (branch,))
    if isinstance(edit, AddPrimaryBranch):
        child, operation = nodes.get(edit.child_id), edit.operation
        if not homogeneous_input_graph(spec, raw_input):
            return MutationRejection(MutationRejectionCode.INPUT_CONTEXT,
                                     "Primary branching needs confirmed homogeneous classification/regression input.")
        if (node is not None or child is None or not ordinary(child.capabilities)
                or not ordinary(operation) or len(child.parent_ids) != 1):
            return invalid
        parents = [nodes[parent_id] for parent_id in child.parent_ids]
        if operation.name in {parent.operation_name for parent in parents}:
            return MutationRejection(MutationRejectionCode.SAME_OPERATION,
                                     "A new primary branch must add a distinct transform.")
        if (raw_input.data_type not in operation.input_data_types
                or operation.output_data_types != ("tabular",)
                or "tabular" not in child.capabilities.input_data_types
                or any(parent.capabilities.kind is not OperationKind.TRANSFORM
                       or parent.capabilities.output_data_types != ("tabular",) for parent in parents)):
            return MutationRejection(MutationRejectionCode.DATA_TYPE,
                                     "Only aligned tabular transform outputs may form a primary feature merge.")
        if operation.requires_target and not raw_input.has_target:
            return MutationRejection(MutationRejectionCode.TARGET_REQUIRED, "The new branch requires a target.")
        branch = GraphNodeSpec(edit.node_id, operation.name, (), operation)
        updated = replace(child, parent_ids=child.parent_ids + (edit.node_id,))
        return replace(spec, nodes=tuple(updated if item is child else item
                                         for item in spec.nodes) + (branch,))
    if isinstance(edit, DropPrimaryTransform):
        if raw_input is None:
            return MutationRejection(MutationRejectionCode.MISSING_CONTEXT,
                                     "Primary removal requires a confirmed homogeneous raw input.")
        if node is None or not drop_primary_allowed(spec, node, raw_input):
            return invalid
        return replace(spec, nodes=tuple(replace(item, parent_ids=())
                                         if node.node_id in item.parent_ids else item
                                         for item in spec.nodes if item is not node))
    if isinstance(edit, DropPrimaryBranch):
        if node is None or not drop_primary_branch_allowed(spec, node, raw_input):
            return invalid
        return replace(spec, nodes=tuple(replace(item, parent_ids=tuple(
            parent_id for parent_id in item.parent_ids if parent_id != node.node_id))
            for item in spec.nodes if item is not node))
    if isinstance(edit, DeduplicateParents):
        if (node is None or not ordinary(node.capabilities)
                or not unary_context_allowed(spec, node.capabilities, raw_input=raw_input)
                or len(set(node.parent_ids)) == len(node.parent_ids)):
            return invalid
        updated = replace(node, parent_ids=tuple(dict.fromkeys(node.parent_ids)))
        return replace(spec, nodes=tuple(updated if item is node else item for item in spec.nodes))
    return invalid
