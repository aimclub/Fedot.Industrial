"""Bounded structural edit enumeration and explicit deterministic selection."""

from __future__ import annotations

from itertools import islice
from typing import Iterator

from fedot_ind.core.optimizer.graph_validation.contracts import (
    GraphSpec, OperationCapabilities, ValidationIssueCode,
)
from fedot_ind.core.optimizer.graph_validation.validation import validate_graph
from fedot_ind.core.optimizer.mutation_planning._graph import (
    apply_edit, budget_excess, candidate_signature, drop_allowed, graph_precondition, graph_signature,
    new_node_id, ordinary, replacement_allowed,
)
from fedot_ind.core.optimizer.mutation_planning.contracts import (
    AppliedMutation, DeduplicateParents, DropTransform, GraphEdit, InsertTransform, MutationCandidates,
    MutationKind, MutationLimits, MutationPlan, MutationRejection,
    MutationRejectionCode, ReplaceOperation, _integer,
)


def apply_mutation_plan(spec: GraphSpec, plan: MutationPlan, *,
                        limits: MutationLimits) -> AppliedMutation | MutationRejection:
    """Apply atomically; rejection never exposes a partially rewritten graph."""
    if graph_signature(spec) != plan.source_signature:
        return MutationRejection(MutationRejectionCode.STALE_PLAN,
                                 "The plan was created for a different graph or context.")
    if len(plan.edits) > limits.max_edits:
        return MutationRejection(MutationRejectionCode.BUDGET_EXCEEDED,
                                 "Plan exceeds the edit budget.")
    normalizations = {edit.node_id for edit in plan.edits if isinstance(edit, DeduplicateParents)}
    rejection = graph_precondition(spec, limits, check_budget=False,
                                   allow_duplicate_parents=bool(normalizations))
    if rejection is not None:
        return rejection
    if max(budget_excess(spec, limits)[:2]) > len(plan.edits):
        return MutationRejection(MutationRejectionCode.BUDGET_EXCEEDED,
                                 "Local edits cannot reduce the graph enough to meet its budgets.")
    original_report = validate_graph(spec)
    locally_normalizable = {ValidationIssueCode.IDENTICAL_PARENT_OPERATIONS,
                            ValidationIssueCode.TOO_MANY_PARENTS}
    if any(issue.code not in locally_normalizable or not set(issue.node_ids) <= normalizations
           for issue in original_report.issues):
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Hard task, device, data and structural errors cannot be rewritten.",
                                 original_report)
    graph = spec
    for edit in plan.edits:
        graph = apply_edit(graph, edit)
        if isinstance(graph, MutationRejection):
            return graph
    if any(budget_excess(graph, limits)):
        return MutationRejection(MutationRejectionCode.BUDGET_EXCEEDED,
                                 "Result exceeds the node, depth or arity budget.")
    report = validate_graph(graph)
    if not report.is_valid:
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Edited graph violates the capability contract.", report)
    return AppliedMutation(graph, report)


def admissible_operations(spec: GraphSpec, node_id: str,
                          operations: tuple[OperationCapabilities, ...], *,
                          limits: MutationLimits) -> tuple[OperationCapabilities, ...]:
    """Replacement subset of the first max_candidates enabled catalog entries.

    The caller supplies only enabled operations. No repositories or global
    catalogs are consulted, and a missing task never gets a fabricated default.
    """
    node = next((node for node in spec.nodes if node.node_id == node_id), None)
    if node is None or graph_precondition(spec, limits) is not None:
        return ()
    signature = graph_signature(spec)
    result = {}
    for operation in islice(operations, limits.max_candidates):
        if replacement_allowed(spec, node, operation):
            plan = MutationPlan(signature, (ReplaceOperation(node_id, operation),))
            if isinstance(apply_mutation_plan(spec, plan, limits=limits), AppliedMutation):
                result.setdefault(operation.name, operation)
    return tuple(result.values())


def _proposals(spec: GraphSpec, operations: tuple[OperationCapabilities, ...],
               kind: MutationKind) -> Iterator[GraphEdit]:
    if kind is MutationKind.REPLACE:
        for node in spec.nodes:
            if ordinary(node.capabilities):
                for operation in operations:
                    yield ReplaceOperation(node.node_id, operation)
    elif kind is MutationKind.ADD:
        for child in spec.nodes:
            if ordinary(child.capabilities):
                for parent_id in child.parent_ids or (None,):
                    for operation in operations:
                        yield InsertTransform(parent_id, child.node_id,
                                              new_node_id(spec, parent_id, child.node_id,
                                                          operation.name), operation)
    elif kind is MutationKind.DROP:
        for node in spec.nodes:
            if drop_allowed(spec, node):
                yield DropTransform(node.node_id)


def enumerate_mutation_plans(
        spec: GraphSpec, operations: tuple[OperationCapabilities, ...], *,
        limits: MutationLimits,
        kinds: tuple[MutationKind, ...] = tuple(MutationKind),
) -> MutationCandidates:
    """Round-robin finite edit streams; at most max_candidates edits are tested.

    Input graph and catalog order are meaningful. Repeated kinds and outcomes
    are deduplicated, while edge/parent order is never permuted. Truncation is
    conservative: reaching the cap means further candidates may exist.
    """
    if not isinstance(operations, tuple) or any(
            not isinstance(operation, OperationCapabilities) for operation in operations):
        raise TypeError("operations must be a tuple of OperationCapabilities")
    if not isinstance(kinds, tuple) or any(not isinstance(kind, MutationKind) for kind in kinds):
        raise TypeError("kinds must be a tuple of MutationKind values")
    rejection = graph_precondition(spec, limits)
    if rejection is not None:
        return MutationCandidates((), 0, False, rejection)
    report = validate_graph(spec)
    if not report.is_valid:
        return MutationCandidates((), 0, False, MutationRejection(
            MutationRejectionCode.INVALID_GRAPH, "Repair the input before mutation planning.",
            report))
    if limits.max_edits == 0:
        return MutationCandidates((), 0, False)
    streams = [_proposals(spec, operations, kind) for kind in dict.fromkeys(kinds)]
    signature = graph_signature(spec)
    plans = []
    seen = {candidate_signature(spec)}
    examined = 0
    while streams and examined < limits.max_candidates:
        remaining = []
        for stream in streams:
            if examined == limits.max_candidates:
                remaining.append(stream)
                continue
            edit = next(stream, None)
            if edit is None:
                continue
            remaining.append(stream)
            examined += 1
            plan = MutationPlan(signature, (edit,))
            outcome = apply_mutation_plan(spec, plan, limits=limits)
            if isinstance(outcome, AppliedMutation):
                key = candidate_signature(outcome.graph)
                if key not in seen:
                    seen.add(key)
                    plans.append(plan)
        streams = remaining
    return MutationCandidates(tuple(plans), examined, bool(streams))


def select_mutation_plan(candidates: MutationCandidates, seed: int) -> MutationPlan | None:
    """Use an explicit seed (e.g. CandidateRandomness.seed), without ambient RNGs."""
    _integer("seed", seed)
    return candidates.plans[seed % len(candidates.plans)] if candidates.plans else None
