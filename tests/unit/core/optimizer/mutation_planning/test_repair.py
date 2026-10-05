from dataclasses import replace

import pytest

from fedot_ind.core.optimizer.graph_validation import (
    ComputeDevice, GraphNodeSpec, GraphSpec, OperationKind, ValidationIssueCode, validate_graph,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, DeduplicateParents, DropTransform, MutationPlan, MutationRejection,
    MutationRejectionCode, RepairedGraph, apply_mutation_plan, graph_depth,
    graph_signature, repair_graph,
)
from .test_planning import LIMITS, capability, chain


def test_valid_repair_is_a_new_equal_graph_and_idempotent():
    spec = chain()
    repaired = repair_graph(spec, limits=LIMITS)
    assert isinstance(repaired, RepairedGraph)
    assert repaired.graph == spec and repaired.graph is not spec
    assert repaired.edits == () and repaired.examined == 0
    assert repair_graph(repaired.graph, limits=LIMITS) == repaired


def test_duplicate_reference_repair_is_order_preserving_and_replayable():
    spec = chain(1)
    spec = replace(spec, nodes=(spec.nodes[0], replace(spec.nodes[1],
                                                       parent_ids=("node-0", "node-0"))))
    snapshot = spec.to_record()
    result = repair_graph(spec, limits=LIMITS)
    assert isinstance(result, RepairedGraph)
    assert result.graph.nodes[-1].parent_ids == ("node-0",)
    assert result.edits == (DeduplicateParents("node-1"),)
    assert result.examined == 1
    replay = apply_mutation_plan(spec, MutationPlan(graph_signature(spec), result.edits), limits=LIMITS)
    assert isinstance(replay, AppliedMutation) and replay.graph == result.graph
    assert spec.to_record() == snapshot
    repeated = repair_graph(result.graph, limits=LIMITS)
    assert repeated.graph == result.graph and repeated.edits == ()


def test_two_distinct_nodes_with_same_operation_are_not_deduplicated():
    transform = capability("same")
    model = capability("model", OperationKind.MODEL, max_parents=2)
    spec = GraphSpec("graph", (GraphNodeSpec("a", transform.name, capabilities=transform),
                               GraphNodeSpec("b", transform.name, capabilities=transform),
                               GraphNodeSpec("root", model.name, ("a", "b"), model)),
                     task_type="classification")
    result = repair_graph(spec, limits=LIMITS)
    assert isinstance(result, MutationRejection) and result.examined == 0
    assert ValidationIssueCode.IDENTICAL_PARENT_OPERATIONS.value in result.report.error_codes


def test_budget_repair_is_bounded_deterministic_and_replayable():
    spec = chain(3)
    limits = replace(LIMITS, max_nodes=2, max_depth=2, max_edits=2)
    result = repair_graph(spec, limits=limits)
    assert isinstance(result, RepairedGraph)
    assert result.edits == (DropTransform("node-1"), DropTransform("node-2"))
    assert len(result.graph.nodes) == graph_depth(result.graph) == 2
    assert result.examined == 2
    assert repair_graph(spec, limits=limits) == result
    replay = apply_mutation_plan(spec, MutationPlan(graph_signature(spec), result.edits), limits=limits)
    assert isinstance(replay, AppliedMutation) and replay.graph == result.graph
    assert repair_graph(result.graph, limits=limits).edits == ()


@pytest.mark.parametrize("changes", [{"max_edits": 1}, {"max_candidates": 1},
                                     {"max_edits": 0}, {"max_candidates": 0}])
def test_repair_failure_is_atomic_and_work_budget_is_total(changes):
    spec = chain(3)
    snapshot = spec.to_record()
    limits = replace(LIMITS, max_nodes=2, max_depth=2, **changes)
    result = repair_graph(spec, limits=limits)
    assert isinstance(result, MutationRejection)
    assert result.code is MutationRejectionCode.REPAIR_EXHAUSTED
    assert result.examined <= limits.max_candidates
    assert not hasattr(result, "graph")
    assert spec.to_record() == snapshot


def test_duplicate_reference_and_drop_share_the_same_edit_budget():
    spec = chain(2)
    spec = replace(spec, nodes=spec.nodes[:-1] + (
        replace(spec.nodes[-1], parent_ids=("node-1", "node-1")),))
    result = repair_graph(spec, limits=replace(LIMITS, max_nodes=2, max_edits=1))
    assert isinstance(result, MutationRejection) and result.examined == 1
    assert result.code is MutationRejectionCode.REPAIR_EXHAUSTED


def test_repair_does_not_drop_type_bridges_or_trim_parent_branches():
    spec = chain(2)
    first = replace(spec.nodes[0], capabilities=replace(spec.nodes[0].capabilities,
                                                        output_data_types=("ts",)))
    bridge = replace(spec.nodes[1], capabilities=replace(spec.nodes[1].capabilities,
                                                         input_data_types=("ts",)))
    spec = replace(spec, nodes=(first, bridge, spec.nodes[2]))
    assert validate_graph(spec).is_valid
    assert isinstance(repair_graph(spec, limits=replace(LIMITS, max_nodes=2)), MutationRejection)
    left, right = capability("left"), capability("right")
    model = capability("model", OperationKind.MODEL, max_parents=None)
    parallel = GraphSpec("parallel", (GraphNodeSpec("a", left.name, capabilities=left),
                                      GraphNodeSpec("b", right.name, capabilities=right),
                                      GraphNodeSpec("root", model.name, ("a", "b"), model)),
                         task_type="classification")
    result = repair_graph(parallel, limits=replace(LIMITS, max_arity=1))
    assert isinstance(result, MutationRejection) and result.examined == 0


def test_repair_preserves_hard_context_and_unknown_errors():
    spec = chain(1)
    invalids = (replace(spec, task_type=None),
                replace(spec, task_type="regression"),
                replace(spec, requested_device=ComputeDevice.CUDA),
                replace(spec, nodes=(replace(spec.nodes[0], capabilities=None), spec.nodes[1])),
                replace(spec, nodes=(replace(spec.nodes[0], parent_ids=("node-1",)), spec.nodes[1])),
                replace(spec, nodes=(replace(spec.nodes[0], parent_ids=("missing",)), spec.nodes[1])))
    for invalid in invalids:
        result = repair_graph(invalid, limits=LIMITS)
        assert isinstance(result, MutationRejection)
        assert result.examined == 0
    mismatched = repair_graph(invalids[1], limits=LIMITS)
    assert ValidationIssueCode.TASK_NOT_SUPPORTED.value in mismatched.report.error_codes
    gpu = repair_graph(invalids[2], limits=LIMITS)
    assert ValidationIssueCode.DEVICE_NOT_SUPPORTED.value in gpu.report.error_codes


def test_repair_does_not_turn_a_transform_root_into_a_model():
    transform = capability("transform")
    spec = GraphSpec("root", (GraphNodeSpec("root", transform.name, capabilities=transform),),
                     task_type="classification")
    result = repair_graph(spec, limits=LIMITS)
    assert isinstance(result, MutationRejection)
    assert ValidationIssueCode.ROOT_NOT_MODEL.value in result.report.error_codes


def test_impossible_deep_input_is_rejected_before_recursive_special_rules():
    spec = chain(1500)
    result = repair_graph(spec, limits=replace(LIMITS, max_nodes=64, max_depth=10))
    assert isinstance(result, MutationRejection)
    assert result.code is MutationRejectionCode.REPAIR_EXHAUSTED and result.examined == 0
