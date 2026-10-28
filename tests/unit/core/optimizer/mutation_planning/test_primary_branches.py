import json
from dataclasses import replace

import pytest

from fedot_ind.core.optimizer.graph_validation import NodePosition, OperationKind, StructuralRole
from fedot_ind.core.optimizer.mutation_planning import (
    AddPrimaryBranch, AppliedMutation, DropPrimaryBranch, MutationKind, MutationPlan, MutationRejection,
    RawInputSpec, ReplaceOperation, apply_mutation_plan, enumerate_mutation_plans, graph_signature,
)
from .test_invariants import independent_graph_oracle
from .test_planning import LIMITS, capability, chain


def feature_graph():
    spec = chain(1)
    return replace(spec, nodes=tuple(replace(node, capabilities=replace(
        node.capabilities, supports_multimodal=True,
        max_parents=2 if node.capabilities.kind is OperationKind.MODEL else 1)) for node in spec.nodes))


RAW = RawInputSpec('tabular', True)
FEATURE = capability('other_feature', supports_multimodal=True)


def apply_branch(spec, operation=FEATURE, raw=RAW, limits=LIMITS):
    plan = MutationPlan(graph_signature(spec), (AddPrimaryBranch('node-1', 'new', operation),), raw)
    return apply_mutation_plan(spec, plan, limits=limits, raw_input=raw)


def test_parallel_features_share_raw_input_not_a_preprocessed_parent():
    source = feature_graph()
    result = apply_branch(source)
    assert isinstance(result, AppliedMutation)
    independent_graph_oracle(result.graph, LIMITS)
    nodes = {node.node_id: node for node in result.graph.nodes}
    assert nodes['node-0'].parent_ids == nodes['new'].parent_ids == ()
    assert nodes['node-1'].parent_ids == ('node-0', 'new')
    assert source == feature_graph()
    encoded = MutationPlan(graph_signature(source), (AddPrimaryBranch('node-1', 'new', FEATURE),), RAW).to_record()
    assert json.loads(json.dumps(encoded)) == encoded
    assert encoded['edits'][0]['kind'] == 'add_primary_branch'


def test_primary_fusion_is_enumerated_and_remains_mutable_with_confirmed_context():
    spec = feature_graph()
    candidates = enumerate_mutation_plans(spec, (FEATURE,), limits=replace(LIMITS, max_candidates=200),
                                          kinds=(MutationKind.ADD,), raw_input=RAW)
    primary = [plan for plan in candidates.plans if isinstance(plan.edits[0], AddPrimaryBranch)]
    assert len(primary) == 1
    outcome = apply_mutation_plan(spec, primary[0], limits=LIMITS, raw_input=RAW)
    root = next(node for node in outcome.graph.nodes if node.node_id == 'node-1')
    new_model = replace(root.capabilities, name='other_model')
    plan = MutationPlan(graph_signature(outcome.graph), (ReplaceOperation('node-1', new_model),), RAW)
    assert isinstance(apply_mutation_plan(outcome.graph, plan, limits=LIMITS, raw_input=RAW), AppliedMutation)
    assert isinstance(apply_mutation_plan(outcome.graph, plan, limits=LIMITS), MutationRejection)


@pytest.mark.parametrize('operation', [
    replace(FEATURE, kind=OperationKind.MODEL),
    replace(FEATURE, output_data_types=('ts',)),
    replace(FEATURE, input_data_types=('image',)),
    replace(FEATURE, allowed_positions=(NodePosition.SECONDARY,)),
    replace(FEATURE, tags=('filtering',)),
    replace(FEATURE, structural_role=StructuralRole.RESAMPLING),
    replace(FEATURE, task_types=('regression',)),
    replace(FEATURE, name='transform-0'),
])
def test_unsafe_feature_operations_are_rejected(operation):
    assert isinstance(apply_branch(feature_graph(), operation), MutationRejection)


def test_fusion_requires_confirmed_input_and_preserves_graph_budgets():
    spec = feature_graph()
    for raw in (None, RawInputSpec('image', True), RawInputSpec('tabular', False)):
        assert isinstance(apply_branch(spec, raw=raw), MutationRejection)
    for limits in (replace(LIMITS, max_arity=1), replace(LIMITS, max_nodes=2)):
        assert isinstance(apply_branch(spec, limits=limits), MutationRejection)
    assert isinstance(apply_branch(replace(spec, task_type='ts_forecasting')), MutationRejection)
    result = apply_branch(spec)
    assert isinstance(apply_branch(result.graph, replace(FEATURE, name='third')), MutationRejection)


def test_model_predictions_are_not_silently_merged_as_features():
    spec = feature_graph()
    previous = replace(spec.nodes[0].capabilities, kind=OperationKind.MODEL)
    spec = replace(spec, nodes=(replace(spec.nodes[0], capabilities=previous), spec.nodes[1]))
    assert isinstance(apply_branch(spec), MutationRejection)


def test_added_branch_can_be_removed_without_dropping_the_last_feature_input():
    spec = feature_graph()
    branched = apply_branch(spec).graph
    candidates = enumerate_mutation_plans(branched, (), limits=LIMITS, kinds=(MutationKind.DROP,), raw_input=RAW)
    drops = [plan for plan in candidates.plans if isinstance(plan.edits[0], DropPrimaryBranch)]
    assert len(drops) == 2
    plan = next(plan for plan in drops if plan.edits[0].node_id == 'new')
    restored = apply_mutation_plan(branched, plan, limits=LIMITS, raw_input=RAW)
    assert isinstance(restored, AppliedMutation)
    assert restored.graph == spec
    last = MutationPlan(graph_signature(restored.graph), (DropPrimaryBranch('node-0'),), RAW)
    assert isinstance(apply_mutation_plan(restored.graph, last, limits=LIMITS, raw_input=RAW), MutationRejection)
