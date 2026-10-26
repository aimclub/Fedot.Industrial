import json
from dataclasses import FrozenInstanceError, replace

import pytest

from fedot_ind.core.optimizer.graph_validation import (
    ComputeDevice, GraphNodeSpec, GraphSpec, NodePosition, OperationCapabilities,
    OperationKind, StructuralRole, ValidationIssueCode, validate_graph,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, DropTransform, InsertTransform, MutationKind, MutationLimits,
    MutationPlan, MutationRejection, MutationRejectionCode, ReplaceOperation,
    admissible_operations, apply_mutation_plan, enumerate_mutation_plans,
    graph_depth, graph_signature, select_mutation_plan,
)


def capability(name, kind=OperationKind.TRANSFORM, **kwargs):
    values = dict(input_data_types=("tabular",), output_data_types=("tabular",),
                  task_types=("classification",), max_parents=1)
    values.update(kwargs)
    return OperationCapabilities(name, kind, **values)


def chain(transforms=2, *, task="classification"):
    nodes = []
    for index in range(transforms):
        op = capability(f"transform-{index}", task_types=(task,))
        nodes.append(GraphNodeSpec(f"node-{index}", op.name,
                                   () if index == 0 else (f"node-{index - 1}",), op))
    model = capability("model", OperationKind.MODEL, task_types=(task,))
    nodes.append(GraphNodeSpec(f"node-{transforms}", model.name,
                               (f"node-{transforms - 1}",) if transforms else (), model))
    return GraphSpec("graph", tuple(nodes), task_type=task)


LIMITS = MutationLimits(max_nodes=16, max_depth=16, max_arity=2)
CATALOG = (capability("alternative-transform"), capability("alternative-model", OperationKind.MODEL))


def apply(spec, edit, limits=LIMITS):
    return apply_mutation_plan(spec, MutationPlan(graph_signature(spec), (edit,)), limits=limits)


def test_all_structural_families_are_present_and_distinct():
    spec = chain()
    snapshot = spec.to_record()
    candidates = enumerate_mutation_plans(spec, CATALOG, limits=LIMITS)
    families = {type(plan.edits[0]) for plan in candidates.plans}
    assert families == {ReplaceOperation, InsertTransform, DropTransform}
    graphs = [apply_mutation_plan(spec, plan, limits=LIMITS).graph for plan in candidates.plans]
    assert len(set(map(graph_signature, graphs))) == len(graphs)
    assert all(graph is not spec and validate_graph(graph).is_valid for graph in graphs)
    assert spec.to_record() == snapshot
    assert candidates.examined <= LIMITS.max_candidates
    assert not candidates.truncated


def test_single_model_gets_safe_primary_addition_and_replacement():
    spec = chain(0)
    candidates = enumerate_mutation_plans(spec, CATALOG, limits=LIMITS)
    additions = [plan for plan in candidates.plans if isinstance(plan.edits[0], InsertTransform)]
    assert len(additions) == 1
    edit = additions[0].edits[0]
    assert edit.parent_id is None and edit.child_id == "node-0"
    outcome = apply_mutation_plan(spec, additions[0], limits=LIMITS)
    assert isinstance(outcome, AppliedMutation)
    assert len(outcome.graph.nodes) == 2
    inserted = next(node for node in outcome.graph.nodes if node.node_id == edit.node_id)
    assert inserted.parent_ids == ()
    assert inserted.capabilities.kind is OperationKind.TRANSFORM
    assert outcome.graph.nodes[0].parent_ids == (inserted.node_id,)
    assert outcome.graph.task_type == spec.task_type
    assert any(isinstance(plan.edits[0], ReplaceOperation) for plan in candidates.plans)
    assert json.loads(json.dumps(additions[0].to_record())) == additions[0].to_record()


def test_primary_addition_rejects_disjoint_implicit_input_and_invented_target():
    old = capability("model", OperationKind.MODEL,
                     input_data_types=("tabular", "ts"), requires_target=False)
    spec = GraphSpec("graph", (GraphNodeSpec("node-0", old.name, capabilities=old),),
                     task_type="classification")
    unsafe = (
        capability("needs-target", input_data_types=("tabular", "ts")),
        capability("disjoint-input", input_data_types=("image",), requires_target=False),
        capability("bad-output", input_data_types=("tabular", "ts"),
                   output_data_types=("image",), requires_target=False),
        capability("only-secondary", input_data_types=("tabular", "ts"),
                   allowed_positions=(NodePosition.SECONDARY,), requires_target=False),
    )
    overlapping = capability("overlapping", requires_target=False)
    safe = capability("safe", input_data_types=("tabular", "ts"), requires_target=False)
    candidates = enumerate_mutation_plans(spec, unsafe + (overlapping, safe), limits=LIMITS,
                                          kinds=(MutationKind.ADD,))
    assert [plan.edits[0].operation.name for plan in candidates.plans] == ["overlapping", "safe"]
    assert isinstance(spec(*InsertTransform(None, "node-0", "new", safe)), AppliedMutation)
    assert isinstance(chain()(*InsertTransform(None, "node-2", "new", safe)), MutationRejection)


def test_replacement_rejects_unsafe_task_role_and_type_changes():
    spec = chain(0)
    catalog = (
        capability("regressor", OperationKind.MODEL, task_types=("regression",)),
        capability("transform"),
        capability("unknown", OperationKind.UNKNOWN),
        capability("source", OperationKind.DATA_SOURCE),
        capability("bad-input", OperationKind.MODEL, input_data_types=("image",)),
        capability("bad-output", OperationKind.MODEL, output_data_types=("image",)),
        capability("special", OperationKind.MODEL, structural_role=StructuralRole.DECOMPOSITION),
        capability("safe", OperationKind.MODEL),
    )
    assert [op.name for op in admissible_operations(spec, "node-0", catalog, limits=LIMITS)] == ["safe"]
    for op in catalog[:-1]:
        assert isinstance(spec(*ReplaceOperation("node-0", op)), MutationRejection)


def test_replacement_accepts_overlapping_primary_type_alternatives():
    spec = chain(0)
    old = replace(spec.nodes[0].capabilities, input_data_types=("ts", "tabular"))
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=old),))
    assert admissible_operations(spec, "node-0", CATALOG, limits=LIMITS) == (CATALOG[1],)
    broad = capability("broad", OperationKind.MODEL,
                       input_data_types=("tabular", "ts", "image"),
                       output_data_types=("tabular", "ts"))
    assert admissible_operations(spec, "node-0", (broad,), limits=LIMITS) == (broad,)


def test_primary_add_accepts_multi_type_normalization_before_tabular_model():
    spec = chain(0)
    normalization = capability("normalization", input_data_types=("tabular", "ts"),
                               output_data_types=("tabular", "ts"))
    candidates = enumerate_mutation_plans(spec, (normalization,), limits=LIMITS,
                                          kinds=(MutationKind.ADD,))
    assert len(candidates.plans) == 1
    assert candidates.plans[0].edits[0].parent_id is None
    outcome = apply_mutation_plan(spec, candidates.plans[0], limits=LIMITS)
    assert isinstance(outcome, AppliedMutation) and outcome.report.is_valid


def test_edge_add_checks_overlapping_alternatives_at_both_endpoints():
    spec = chain(1)
    scaling = replace(spec.nodes[0].capabilities, input_data_types=("tabular", "ts"),
                      output_data_types=("tabular", "ts"))
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=scaling), spec.nodes[1]))
    normalization = capability("normalization", output_data_types=("tabular", "ts"))
    outcome = spec(*InsertTransform("node-0", "node-1", "new", normalization))
    assert isinstance(outcome, AppliedMutation) and outcome.report.is_valid
    for incompatible in (replace(normalization, input_data_types=("image",)),
                         replace(normalization, output_data_types=("image",))):
        assert isinstance(spec(*InsertTransform("node-0", "node-1", "new", incompatible)),
                          MutationRejection)


def test_secondary_replacement_accepts_union_types_but_final_edges_are_validated():
    spec = chain(2)
    broad = capability("broad", input_data_types=("tabular", "ts"),
                       output_data_types=("tabular", "ts"))
    outcome = spec(*ReplaceOperation("node-1", broad))
    assert isinstance(outcome, AppliedMutation) and outcome.report.is_valid
    # The old/new contracts overlap on ts, but the actual child requires tabular.
    old = replace(spec.nodes[1].capabilities, output_data_types=("tabular", "ts"))
    spec = replace(spec, nodes=(spec.nodes[0], replace(spec.nodes[1], capabilities=old), spec.nodes[2]))
    outcome = spec(*ReplaceOperation("node-1", replace(broad, output_data_types=("ts",))))
    assert isinstance(outcome, MutationRejection)
    assert ValidationIssueCode.DATA_TYPE_MISMATCH.value in outcome.report.error_codes


def test_capability_position_device_serialization_and_arity_constraints():
    spec = replace(chain(), requested_device=ComputeDevice.CPU, require_serializable=True)
    catalog = (
        capability("gpu", OperationKind.MODEL, devices=(ComputeDevice.CUDA,)),
        capability("not-serializable", OperationKind.MODEL, serializable=False),
        capability("primary-only", OperationKind.MODEL, allowed_positions=(NodePosition.PRIMARY,)),
        capability("no-parents", OperationKind.MODEL, max_parents=0),
        CATALOG[1],
    )
    assert admissible_operations(spec, "node-2", catalog, limits=LIMITS) == (CATALOG[1],)


def test_add_and_drop_are_checked_against_data_flow():
    spec = chain()
    bad = capability("image", input_data_types=("image",))
    assert isinstance(spec(*InsertTransform("node-0", "node-1", "new", bad)), MutationRejection)
    first = replace(spec.nodes[0].capabilities, output_data_types=("ts",))
    bridge = replace(spec.nodes[1].capabilities, input_data_types=("ts",))
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=first),
                                replace(spec.nodes[1], capabilities=bridge), spec.nodes[2]))
    assert validate_graph(spec).is_valid
    outcome = spec(*DropTransform("node-1"))
    assert isinstance(outcome, MutationRejection)
    assert ValidationIssueCode.DATA_TYPE_MISMATCH.value in outcome.report.error_codes
    assert isinstance(spec(*DropTransform("node-0")), MutationRejection)
    assert isinstance(spec(*DropTransform("node-2")), MutationRejection)


def test_drop_preserves_parent_order_and_shared_branch():
    spec = chain()
    third = capability("third")
    root = replace(spec.nodes[2], parent_ids=("other", "node-1"),
                   capabilities=replace(spec.nodes[2].capabilities, max_parents=2))
    spec = replace(spec, nodes=(spec.nodes[0], spec.nodes[1], root,
                                GraphNodeSpec("other", third.name, ("node-0",), third)))
    outcome = spec(*DropTransform("node-1"))
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes[1].parent_ids == ("other", "node-0")
    assert outcome.graph.nodes[2].parent_ids == ("node-0",)
    assert "node-1" not in outcome.graph.node_ids


@pytest.mark.parametrize("field,value", [("max_nodes", 0), ("max_depth", -1),
                                         ("max_candidates", False), ("max_edits", 1.5),
                                         ("max_arity", -1)])
def test_limits_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(LIMITS, **{field: value})


@pytest.mark.parametrize("field", ["max_nodes", "max_depth"])
def test_add_respects_node_and_depth_budget(field):
    spec = chain(0)
    limits = replace(LIMITS, **{field: 1})
    candidates = enumerate_mutation_plans(spec, CATALOG, limits=limits, kinds=(MutationKind.ADD,))
    assert candidates.plans == ()
    assert apply(spec, InsertTransform(None, "node-0", "new", CATALOG[0]), limits).code is \
        MutationRejectionCode.BUDGET_EXCEEDED


def test_global_arity_budget_applies_beyond_capability_limit():
    left, right = capability("left"), capability("right")
    model = capability("model", OperationKind.MODEL, max_parents=None)
    spec = GraphSpec("graph", (GraphNodeSpec("a", left.name, capabilities=left),
                               GraphNodeSpec("b", right.name, capabilities=right),
                               GraphNodeSpec("root", model.name, ("a", "b"), model)),
                     task_type="classification")
    assert validate_graph(spec).is_valid
    result = enumerate_mutation_plans(spec, CATALOG, limits=replace(LIMITS, max_arity=1))
    assert result.rejection.code is MutationRejectionCode.BUDGET_EXCEEDED
    assert result.examined == 0


def test_candidate_cap_is_total_and_round_robin_includes_each_family():
    spec = chain()
    catalog = tuple(capability(f"candidate-{index}") for index in range(100))
    limited = enumerate_mutation_plans(spec, catalog, limits=replace(LIMITS, max_candidates=3))
    assert limited.examined == 3 and limited.truncated
    assert {type(plan.edits[0]) for plan in limited.plans} == \
        {ReplaceOperation, InsertTransform, DropTransform}
    zero = enumerate_mutation_plans(spec, catalog, limits=replace(LIMITS, max_candidates=0))
    assert zero.examined == 0 and zero.plans == ()
    disabled = enumerate_mutation_plans(spec, catalog, limits=replace(LIMITS, max_edits=0))
    assert disabled.examined == 0 and disabled.plans == ()


def test_explicit_seed_and_input_order_are_deterministic_and_do_not_consume_rngs():
    import random
    import numpy as np

    python_state, numpy_state = random.getstate(), np.random.get_state()
    spec = chain()
    first = enumerate_mutation_plans(spec, CATALOG, limits=LIMITS)
    assert first == enumerate_mutation_plans(spec, CATALOG, limits=LIMITS)
    for seed, expected in enumerate(first.plans):
        assert select_mutation_plan(first, seed) == expected
        assert select_mutation_plan(first, seed + len(first.plans)) == expected
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    with pytest.raises(ValueError):
        select_mutation_plan(first, True)
    with pytest.raises(ValueError):
        select_mutation_plan(first, -1)


def test_stale_plans_reject_context_and_graph_changes():
    spec = chain()
    plan = enumerate_mutation_plans(spec, CATALOG, limits=LIMITS).plans[0]
    for changed in (replace(spec, task_type="regression"),
                    replace(spec, requested_device=ComputeDevice.CUDA),
                    replace(spec, nodes=tuple(reversed(spec.nodes)))):
        assert apply_mutation_plan(changed, plan, limits=LIMITS).code is MutationRejectionCode.STALE_PLAN
    with pytest.raises(FrozenInstanceError):
        plan.edits = ()
    with pytest.raises(FrozenInstanceError):
        plan.edits[0].node_id = "changed"
    with pytest.raises(ValueError):
        MutationPlan("signature", [])
    with pytest.raises(TypeError):
        ReplaceOperation("node-0", {})


def test_enabled_catalog_only_and_repeated_entries_do_not_duplicate_plans():
    spec = chain(0)
    candidates = enumerate_mutation_plans(spec, (CATALOG[1], CATALOG[1]), limits=LIMITS)
    assert len(candidates.plans) == 1
    assert candidates.plans[0].edits[0].operation is CATALOG[1]
    assert enumerate_mutation_plans(spec, (), limits=LIMITS).plans == ()
    assert enumerate_mutation_plans(spec, CATALOG, limits=LIMITS, kinds=()).plans == ()
    with pytest.raises(TypeError):
        enumerate_mutation_plans(spec, list(CATALOG), limits=LIMITS)
    with pytest.raises(TypeError):
        enumerate_mutation_plans(spec, CATALOG, limits=LIMITS, kinds=("connect",))


def test_equal_structures_with_different_inserted_ids_are_deduplicated():
    spec = chain(1)
    transform = spec.nodes[0].capabilities
    candidates = enumerate_mutation_plans(spec, (transform,), limits=LIMITS,
                                          kinds=(MutationKind.ADD,))
    # Insertion before/after this transform yields the same ordered A -> A -> model graph.
    assert candidates.examined == 3
    assert len(candidates.plans) == 1


def test_missing_context_unknowns_cycles_and_hard_device_errors_do_not_get_rewritten():
    spec = chain(0)
    cases = (replace(spec, task_type=None),
             replace(spec, nodes=(replace(spec.nodes[0], capabilities=None),)),
             replace(spec, nodes=(replace(spec.nodes[0], parent_ids=("node-0",)),)),
             replace(spec, requested_device=ComputeDevice.CUDA))
    for invalid in cases:
        result = enumerate_mutation_plans(invalid, CATALOG, limits=LIMITS)
        assert result.rejection is not None and not result.plans and result.examined == 0
    gpu = replace(CATALOG[1], devices=(ComputeDevice.CUDA,))
    outcome = cases[-1](*ReplaceOperation("node-0", gpu))
    assert isinstance(outcome, MutationRejection)
    assert ValidationIssueCode.DEVICE_NOT_SUPPORTED.value in outcome.report.error_codes


def test_multimodal_sources_and_structural_roles_are_conservatively_preserved():
    source = capability("source", OperationKind.DATA_SOURCE)
    model = capability("multi", OperationKind.MODEL, supports_multimodal=True)
    spec = GraphSpec("graph", (GraphNodeSpec("source", source.name, capabilities=source),
                               GraphNodeSpec("root", model.name, ("source",), model)),
                     task_type="classification")
    assert validate_graph(spec).is_valid
    assert enumerate_mutation_plans(spec, CATALOG, limits=LIMITS).plans == ()
    assert isinstance(spec(*ReplaceOperation("source", CATALOG[0])), MutationRejection)
    special = replace(chain().nodes[1].capabilities, structural_role=StructuralRole.RESAMPLING)
    protected = replace(chain(), nodes=(chain().nodes[0],
                                        replace(chain().nodes[1], capabilities=special), chain().nodes[2]))
    assert isinstance(protected(*DropTransform("node-1")), MutationRejection)


def test_depth_is_node_count_and_detects_ambiguous_graphs():
    assert graph_depth(chain(0)) == 1 and graph_depth(chain(4)) == 5
    assert graph_depth(GraphSpec("empty", ())) == 0
    node = chain(0).nodes[0]
    assert graph_depth(replace(chain(0), nodes=(replace(node, parent_ids=(node.node_id,)),))) is None
    assert graph_depth(replace(chain(0), nodes=(node, node))) is None
    assert graph_depth(replace(chain(0), nodes=(replace(node, parent_ids=("missing",)),))) is None
