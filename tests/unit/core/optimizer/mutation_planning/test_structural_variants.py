import json
from dataclasses import FrozenInstanceError, replace

from hypothesis import given, settings, strategies as st
import pytest

from fedot_ind.core.optimizer.graph_validation import (
    GraphNodeSpec, GraphSpec, OperationKind, StructuralRole, ValidationIssueCode, validate_graph,
)
from fedot_ind.core.optimizer.mutation_planning import (
    AddBranch, AppliedMutation, DropPrimaryTransform, DropTransform, InsertTransform,
    MutationKind, MutationLimits, MutationPlan, MutationRejection, MutationRejectionCode,
    RawInputSpec, RepairedGraph, ReplaceOperation, apply_mutation_plan,
    enumerate_mutation_plans, graph_depth, graph_signature, repair_graph, select_mutation_plan,
)
from .test_invariants import independent_graph_oracle
from .test_planning import LIMITS, capability, chain


RAW = RawInputSpec("tabular", True)


def branch_graph():
    spec = chain(1)
    root = replace(spec.nodes[-1], capabilities=replace(spec.nodes[-1].capabilities, max_parents=3))
    return replace(spec, nodes=spec.nodes[:-1] + (root,))


def apply_variant(spec, edit, *, raw_input=None, limits=LIMITS):
    plan = MutationPlan(graph_signature(spec), (edit,), raw_input)
    return apply_mutation_plan(spec, plan, limits=limits, raw_input=raw_input)


@pytest.mark.parametrize("kind", [OperationKind.TRANSFORM, OperationKind.MODEL])
def test_branch_reuses_existing_input_and_preserves_direct_edge(kind):
    spec = branch_graph()
    snapshot = spec.to_record()
    operation = capability("branch", kind)
    edit = AddBranch("node-0", "node-1", "branch", operation)
    outcome = apply_variant(spec, edit)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes[1].parent_ids == ("node-0", "branch")
    assert outcome.graph.nodes[-1].parent_ids == ("node-0",)
    assert outcome.graph.nodes[0] is spec.nodes[0]
    assert len(outcome.graph.nodes) == 3 and graph_depth(outcome.graph) == 3
    assert independent_graph_oracle(outcome.graph, LIMITS) == 3
    assert spec.to_record() == snapshot
    record = MutationPlan(graph_signature(spec), (edit,)).to_record()
    assert record["edits"][0]["kind"] == "add_branch"
    assert json.loads(json.dumps(record)) == record
    with pytest.raises(FrozenInstanceError):
        edit.parent_id = "changed"


def test_add_enumeration_interleaves_insertion_and_branch_under_one_budget():
    spec = branch_graph()
    operations = tuple(capability(f"candidate-{index}") for index in range(200))
    limits = replace(LIMITS, max_candidates=2)
    candidates = enumerate_mutation_plans(spec, operations, limits=limits, kinds=(MutationKind.ADD,))
    assert candidates.examined == 2 and candidates.truncated
    assert {type(plan.edits[0]) for plan in candidates.plans} == {InsertTransform, AddBranch}
    assert candidates == enumerate_mutation_plans(spec, operations, limits=limits, kinds=(MutationKind.ADD,))


@pytest.mark.parametrize("changes", [{"max_arity": 1}, {"max_nodes": 2},
                                     {"max_depth": 2}, {"max_edits": 0}])
def test_branch_respects_every_resource_budget(changes):
    spec = branch_graph()
    result = apply_variant(spec, AddBranch("node-0", "node-1", "branch", capability("branch")),
                           limits=replace(LIMITS, **changes))
    assert isinstance(result, MutationRejection)
    assert result.code is MutationRejectionCode.BUDGET_EXCEEDED


def test_branch_capability_arity_is_checked_independently_of_global_budget():
    result = apply_variant(chain(1), AddBranch("node-0", "node-1", "branch", capability("branch")))
    assert isinstance(result, MutationRejection)
    assert ValidationIssueCode.TOO_MANY_PARENTS.value in result.report.error_codes


@pytest.mark.parametrize("parent,child,new", [("node-0", "node-2", "branch"),
                                              ("missing", "node-1", "branch"),
                                              ("node-0", "missing", "branch"),
                                              ("node-0", "node-1", "node-0")])
def test_branch_rejects_nonlocal_missing_and_cyclic_coordinates(parent, child, new):
    assert isinstance(apply_variant(chain(2), AddBranch(parent, child, new, capability("branch"))),
                      MutationRejection)


def test_branch_requires_common_type_across_all_merge_parents():
    a = capability("a")
    b = capability("b", output_data_types=("tabular", "ts"))
    model = capability("model", OperationKind.MODEL, input_data_types=("tabular", "ts"), max_parents=3)
    spec = GraphSpec("merge", (GraphNodeSpec("a", a.name, capabilities=a),
                               GraphNodeSpec("b", b.name, capabilities=b),
                               GraphNodeSpec("root", model.name, ("a", "b"), model)),
                     task_type="classification")
    assert validate_graph(spec).is_valid
    incompatible = capability("branch", output_data_types=("ts",))
    assert isinstance(apply_variant(spec, AddBranch("b", "root", "branch", incompatible)), MutationRejection)
    compatible = replace(incompatible, output_data_types=("tabular", "ts"))
    limits = replace(LIMITS, max_arity=3)
    outcome = apply_variant(spec, AddBranch("b", "root", "branch", compatible), limits=limits)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes[2].parent_ids == ("a", "b", "branch")


@pytest.mark.parametrize("values", [{"task_types": ("regression",)},
                                    {"input_data_types": ("image",)},
                                    {"output_data_types": ("image",)},
                                    {"structural_role": StructuralRole.RESAMPLING},
                                    {"supports_multimodal": True}, {"tags": ("filtering",)}])
def test_branch_rejects_incompatible_or_context_sensitive_operations(values):
    operation = capability("branch", **values)
    assert isinstance(apply_variant(branch_graph(), AddBranch("node-0", "node-1", "branch", operation)),
                      MutationRejection)


def test_primary_drop_requires_confirmed_raw_input_and_does_not_invent_it():
    spec = chain(1)
    snapshot = spec.to_record()
    assert enumerate_mutation_plans(spec, (), limits=LIMITS, kinds=(MutationKind.DROP,)).plans == ()
    assert apply_variant(spec, DropPrimaryTransform("node-0")).code is MutationRejectionCode.MISSING_CONTEXT
    candidates = enumerate_mutation_plans(spec, (), limits=LIMITS, kinds=(MutationKind.DROP,), raw_input=RAW)
    assert len(candidates.plans) == 1 and candidates.plans[0].raw_input == RAW
    plan = candidates.plans[0]
    outcome = apply_mutation_plan(spec, plan, limits=LIMITS, raw_input=RAW)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes == (replace(spec.nodes[1], parent_ids=()),)
    assert outcome.graph.task_type == spec.task_type and outcome.graph.graph_id == spec.graph_id
    assert plan.to_record()["raw_input"] == {"data_type": "tabular", "has_target": True}
    assert plan.to_record()["edits"] == [{"kind": "drop_primary", "node_id": "node-0"}]
    assert spec.to_record() == snapshot
    for raw_input in (None, RawInputSpec("ts", True), RawInputSpec("tabular", False)):
        assert apply_mutation_plan(spec, plan, limits=LIMITS, raw_input=raw_input).code is \
            MutationRejectionCode.STALE_PLAN


def test_primary_drop_checks_actual_raw_type_not_just_old_output_alternatives():
    spec = chain(1)
    transform = replace(spec.nodes[0].capabilities, input_data_types=("ts",))
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=transform), spec.nodes[1]))
    assert validate_graph(spec).is_valid
    assert isinstance(apply_variant(spec, DropPrimaryTransform("node-0"),
                                    raw_input=RawInputSpec("ts", True)), MutationRejection)
    assert isinstance(apply_variant(spec, DropPrimaryTransform("node-0"), raw_input=RAW), MutationRejection)


def test_forecast_drop_preserves_raw_ts_and_the_lagged_type_bridge():
    smoothing = capability("smoothing", input_data_types=("ts",), output_data_types=("ts",),
                           task_types=("ts_forecasting",))
    lagged = capability("lagged", input_data_types=("ts",), task_types=("ts_forecasting",))
    model = capability("ridge", OperationKind.MODEL, task_types=("ts_forecasting",))
    spec = GraphSpec("forecast", (GraphNodeSpec("smooth", smoothing.name, capabilities=smoothing),
                                  GraphNodeSpec("lagged", lagged.name, ("smooth",), lagged),
                                  GraphNodeSpec("root", model.name, ("lagged",), model)),
                     task_type="ts_forecasting")
    raw_input = RawInputSpec("ts", True)
    outcome = apply_variant(spec, DropPrimaryTransform("smooth"), raw_input=raw_input)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes[0].parent_ids == () and outcome.graph.nodes[1].parent_ids == ("lagged",)
    assert isinstance(apply_variant(spec, DropTransform("lagged"), raw_input=raw_input), MutationRejection)


def test_primary_drop_does_not_trim_a_merge_branch():
    spec = branch_graph()
    other = capability("other")
    spec = replace(spec, nodes=(spec.nodes[0], replace(spec.nodes[1], parent_ids=("node-0", "other")),
                                GraphNodeSpec("other", other.name, capabilities=other)))
    assert validate_graph(spec).is_valid
    assert isinstance(apply_variant(spec, DropPrimaryTransform("node-0"), raw_input=RAW), MutationRejection)


def test_primary_drop_can_rebind_shared_unary_children_to_the_same_known_input():
    spec = chain(1)
    a, b = capability("a"), capability("b")
    root = replace(spec.nodes[-1], parent_ids=("a", "b"),
                   capabilities=replace(spec.nodes[-1].capabilities, max_parents=2))
    spec = replace(spec, nodes=(spec.nodes[0], GraphNodeSpec("a", a.name, ("node-0",), a),
                                GraphNodeSpec("b", b.name, ("node-0",), b), root))
    outcome = apply_variant(spec, DropPrimaryTransform("node-0"), raw_input=RAW)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes[0].parent_ids == outcome.graph.nodes[1].parent_ids == ()
    assert outcome.graph.nodes[-1].parent_ids == ("a", "b")


def test_raw_context_enforces_known_type_and_target_for_other_mutations_too():
    spec = chain(0)
    old = replace(spec.nodes[0].capabilities, input_data_types=("tabular", "ts"))
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=old),))
    only_ts = capability("only-ts", OperationKind.MODEL, input_data_types=("ts",))
    assert isinstance(apply_variant(spec, ReplaceOperation("node-0", only_ts)), AppliedMutation)
    assert isinstance(apply_variant(spec, ReplaceOperation("node-0", only_ts), raw_input=RAW), MutationRejection)
    assert isinstance(repair_graph(spec, limits=LIMITS, raw_input=RawInputSpec("tabular", False)),
                      MutationRejection)
    with pytest.raises(ValueError):
        RawInputSpec("", True)
    with pytest.raises(TypeError):
        RawInputSpec("tabular", 1)
    with pytest.raises(TypeError):
        MutationPlan(graph_signature(spec), (DropPrimaryTransform("node-0"),), {})


def test_multimodal_capability_is_editable_on_a_generic_single_input_chain():
    quantile = capability("quantile_torch", input_data_types=("tabular",),
                          output_data_types=("tensor",), supports_multimodal=True)
    pdl = capability("pdl", OperationKind.MODEL, input_data_types=("tensor",), supports_multimodal=True)
    spec = GraphSpec("unary-torch", (GraphNodeSpec("quantile", quantile.name, capabilities=quantile),
                                     GraphNodeSpec("root", pdl.name, ("quantile",), pdl)),
                     task_type="classification")
    operations = (replace(quantile, name="quantile-alt"), replace(pdl, name="pdl-alt"),
                  capability("tensor-scale", input_data_types=("tensor",),
                             output_data_types=("tensor",), supports_multimodal=True))
    candidates = enumerate_mutation_plans(spec, operations, limits=LIMITS)
    assert candidates.rejection is None and candidates.plans
    assert any(isinstance(plan.edits[0], ReplaceOperation) for plan in candidates.plans)
    assert any(isinstance(plan.edits[0], InsertTransform) for plan in candidates.plans)
    assert all(not isinstance(plan.edits[0], AddBranch) for plan in candidates.plans)
    for plan in candidates.plans:
        outcome = apply_mutation_plan(spec, plan, limits=LIMITS)
        assert isinstance(outcome, AppliedMutation)
        independent_graph_oracle(outcome.graph, LIMITS)
    assert isinstance(repair_graph(spec, limits=LIMITS, raw_input=RAW), RepairedGraph)


def test_image_input_quantile_to_pdl_has_replace_and_primary_drop_without_add():
    quantile = capability("quantile_torch", input_data_types=("image", "tabular", "ts"),
                          supports_multimodal=True, requires_target=False)
    pdl = capability("pdl", OperationKind.MODEL, input_data_types=("image", "tabular"),
                     supports_multimodal=False)
    spec = GraphSpec("image-profile", (GraphNodeSpec("quantile", quantile.name, capabilities=quantile),
                                       GraphNodeSpec("root", pdl.name, ("quantile",), pdl)),
                     task_type="classification")
    operations = (replace(quantile, name="quantile-alt"), pdl)
    raw_input = RawInputSpec("image", True)
    kinds = (MutationKind.REPLACE, MutationKind.DROP)
    candidates = enumerate_mutation_plans(spec, operations, limits=LIMITS,
                                          kinds=kinds, raw_input=raw_input)
    assert {type(plan.edits[0]) for plan in candidates.plans} == {ReplaceOperation, DropPrimaryTransform}
    drop = next(plan for plan in candidates.plans if isinstance(plan.edits[0], DropPrimaryTransform))
    outcome = apply_mutation_plan(spec, drop, limits=LIMITS, raw_input=raw_input)
    assert isinstance(outcome, AppliedMutation)
    assert outcome.graph.nodes == (replace(spec.nodes[1], parent_ids=()),)
    incompatible_raw = RawInputSpec("ts", True)
    ts_candidates = enumerate_mutation_plans(spec, operations, limits=LIMITS,
                                             kinds=kinds, raw_input=incompatible_raw)
    assert ts_candidates.rejection is None and ts_candidates.plans
    assert all(isinstance(plan.edits[0], ReplaceOperation) for plan in ts_candidates.plans)


def test_multimodal_capability_allows_unary_drop_and_known_primary_drop():
    spec = chain(2)
    spec = replace(spec, nodes=tuple(replace(node, capabilities=replace(
        node.capabilities, supports_multimodal=True)) for node in spec.nodes))
    assert isinstance(apply_variant(spec, DropTransform("node-1")), AppliedMutation)
    assert isinstance(apply_variant(spec, DropPrimaryTransform("node-0"), raw_input=RAW), AppliedMutation)
    root = replace(spec.nodes[-1], capabilities=replace(spec.nodes[-1].capabilities, max_parents=3))
    spec = replace(spec, nodes=spec.nodes[:-1] + (root,))
    assert isinstance(apply_variant(spec, AddBranch("node-1", "node-2", "branch", capability("branch"))),
                      MutationRejection)


def test_multimodal_capability_does_not_unlock_actual_source_merge_or_special_contexts():
    a, b = capability("a"), capability("b")
    model = capability("multi", OperationKind.MODEL, max_parents=3, supports_multimodal=True)
    spec = GraphSpec("merge", (GraphNodeSpec("a", a.name, capabilities=a),
                               GraphNodeSpec("b", b.name, capabilities=b),
                               GraphNodeSpec("root", model.name, ("a", "b"), model)),
                     task_type="classification")
    alternative = replace(model, name="alternative")
    assert isinstance(apply_variant(spec, ReplaceOperation("root", alternative)), MutationRejection)
    assert isinstance(apply_variant(spec, AddBranch("a", "root", "branch", capability("branch"))),
                      MutationRejection)
    # An explicit homogeneous input distinguishes feature fusion from multiple data sources.
    assert isinstance(repair_graph(spec, limits=LIMITS, raw_input=RAW), RepairedGraph)
    source = replace(a, kind=OperationKind.DATA_SOURCE)
    source_spec = GraphSpec("source", (GraphNodeSpec("a", source.name, capabilities=source),
                                       GraphNodeSpec("root", model.name, ("a",), model)),
                            task_type="classification")
    assert isinstance(apply_variant(source_spec, ReplaceOperation("root", alternative)), MutationRejection)
    assert isinstance(repair_graph(source_spec, limits=LIMITS, raw_input=RAW), MutationRejection)
    special = replace(a, structural_role=StructuralRole.RESAMPLING)
    special_spec = replace(source_spec, nodes=(GraphNodeSpec("a", special.name, capabilities=special),
                                               source_spec.nodes[1]))
    assert validate_graph(special_spec).is_valid
    assert isinstance(apply_variant(special_spec, ReplaceOperation("root", alternative)), MutationRejection)


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(("tabular", "ts", "image")),
       st.sampled_from((OperationKind.TRANSFORM, OperationKind.MODEL)), st.integers(0, 1000))
def test_generated_branch_has_independent_sharing_depth_and_budget_oracle(data_type, kind, seed):
    source = capability("source", input_data_types=(data_type,), output_data_types=(data_type,))
    model = capability("model", OperationKind.MODEL, input_data_types=(data_type,), max_parents=2)
    spec = GraphSpec("generated", (GraphNodeSpec("source", source.name, capabilities=source),
                                   GraphNodeSpec("root", model.name, ("source",), model)),
                     task_type="classification")
    operation = capability("branch", kind, input_data_types=(data_type,), output_data_types=(data_type,))
    candidates = enumerate_mutation_plans(spec, (operation,), limits=LIMITS, kinds=(MutationKind.ADD,))
    branches = [plan for plan in candidates.plans if isinstance(plan.edits[0], AddBranch)]
    assert len(branches) == 1
    assert select_mutation_plan(candidates, seed) == select_mutation_plan(candidates, seed)
    outcome = apply_mutation_plan(spec, branches[0], limits=LIMITS)
    assert isinstance(outcome, AppliedMutation)
    independent_graph_oracle(outcome.graph, LIMITS)
    edit = branches[0].edits[0]
    assert outcome.graph.nodes[1].parent_ids == ("source", edit.node_id)
    assert outcome.graph.nodes[-1].parent_ids == ("source",)
    assert len(outcome.graph.node_ids) == len(set(outcome.graph.node_ids)) == 3


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 8), st.integers(0, 8), st.integers(0, 8), st.integers(0, 8))
def test_generated_known_raw_repair_has_exact_fuel_oracle(count, removal, edit_cap, candidate_cap):
    spec = chain(count)
    removal %= count + 1
    limits = MutationLimits(count + 1 - removal, count + 1 - removal,
                            max_edits=edit_cap, max_candidates=candidate_cap, max_arity=1)
    snapshot = spec.to_record()
    result = repair_graph(spec, limits=limits, raw_input=RAW)
    if removal <= min(edit_cap, candidate_cap):
        assert isinstance(result, RepairedGraph)
        assert result.examined == len(result.edits) == removal
        assert len(result.graph.nodes) == count + 1 - removal
        independent_graph_oracle(result.graph, limits)
        again = repair_graph(result.graph, limits=limits, raw_input=RAW)
        assert again.graph == result.graph and again.edits == () and again.examined == 0
        if result.edits:
            plan = MutationPlan(graph_signature(spec), result.edits, RAW)
            replay = apply_mutation_plan(spec, plan, limits=limits, raw_input=RAW)
            assert isinstance(replay, AppliedMutation) and replay.graph == result.graph
    else:
        assert isinstance(result, MutationRejection)
    assert result.examined <= candidate_cap
    assert spec.to_record() == snapshot
