"""Bounded, deterministic local repairs; no task or data adaptation."""

from __future__ import annotations

from dataclasses import replace

from fedot_ind.core.optimizer.graph_validation.contracts import GraphSpec
from fedot_ind.core.optimizer.graph_validation.validation import validate_graph
from fedot_ind.core.optimizer.mutation_planning._graph import (
    apply_edit, budget_excess, drop_allowed, graph_precondition,
)
from fedot_ind.core.optimizer.mutation_planning.contracts import (
    DeduplicateParents, DropTransform, GraphEdit, MutationLimits, MutationRejection,
    MutationRejectionCode, RepairedGraph,
)


def repair_graph(spec: GraphSpec, *, limits: MutationLimits) -> RepairedGraph | MutationRejection:
    """Normalize repeated edge references, then greedily bypass unary transforms.

    Bypasses must strictly reduce the total budget excess and pass the existing
    validator. At most max_edits rewrites and max_candidates proposals are used
    across the entire repair, not per step. A failure is atomic. Source nodes,
    special/multimodal operations, cycles and incompatible task/data contracts
    are never repaired by inventing input, changing roles or removing branches.
    """
    rejection = graph_precondition(spec, limits, check_budget=False, allow_duplicate_parents=True)
    if rejection is not None:
        return rejection
    # A unary bypass removes one node and reduces any path by at most one.
    if max(budget_excess(spec, limits)[:2]) > min(limits.max_edits, limits.max_candidates):
        return MutationRejection(MutationRejectionCode.REPAIR_EXHAUSTED,
                                 "The graph requires more local edits than the work budget permits.")
    graph = replace(spec)
    edits: list[GraphEdit] = []
    examined = 0

    def exhausted() -> MutationRejection:
        return MutationRejection(MutationRejectionCode.REPAIR_EXHAUSTED,
                                 "No safe local repair remains within the work budget.",
                                 examined=examined)

    for node in spec.nodes:
        if len(set(node.parent_ids)) != len(node.parent_ids):
            if len(edits) >= limits.max_edits or examined >= limits.max_candidates:
                return exhausted()
            examined += 1
            edit = DeduplicateParents(node.node_id)
            updated = apply_edit(graph, edit)
            if isinstance(updated, MutationRejection):
                return replace(updated, examined=examined)
            edits.append(edit)
            graph = updated
    report = validate_graph(graph)
    if not report.is_valid:
        return MutationRejection(MutationRejectionCode.INVALID_GRAPH,
                                 "Task, data, root, arity or special-operation repair is unsupported.",
                                 report, examined)
    while any(budget_excess(graph, limits)):
        if len(edits) >= limits.max_edits or examined >= limits.max_candidates:
            return exhausted()
        old_excess = sum(budget_excess(graph, limits))
        accepted = False
        for node in graph.nodes:
            if not drop_allowed(graph, node):
                continue
            if examined >= limits.max_candidates:
                return exhausted()
            examined += 1
            edit = DropTransform(node.node_id)
            updated = apply_edit(graph, edit)
            if isinstance(updated, MutationRejection):
                continue
            if sum(budget_excess(updated, limits)) >= old_excess:
                continue
            candidate_report = validate_graph(updated)
            if candidate_report.is_valid:
                graph, report = updated, candidate_report
                edits.append(edit)
                accepted = True
                break
        if not accepted:
            return exhausted()
    return RepairedGraph(graph, tuple(edits), examined, report)
