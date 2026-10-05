from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.core.optimizer.graph_validation import NodePosition
from fedot_ind.core.optimizer.mutation_runtime import (
    MutationExecutionStatus, PlannedMutation,
    advisor_allows_plan, interpret_graph_spec, load_runtime_operation_catalog, project_for_mutation,
)
from fedot_ind.core.optimizer.configuration import MutationPlanningPolicy
from fedot_ind.core.optimizer.mutation import mutation_weights
from fedot_ind.core.optimizer.randomness import operator_random_scope
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.opt_history_objects.individual import Individual

from benchmark.industrial.experiments.mutation_planning.runner import _environment


def _planned(actions=None, *, policy=None, verifier=None):
    settings = {"max_depth": 5, "max_arity": 3}
    scenario = {"name": "test", "task": "regression", "operations": ["scaling", "ridge", "dtreg", "normalization"],
                "initial_chain": ["scaling", "ridge"]}
    requirements, params, graph = _environment(scenario, settings)
    if verifier is not None:
        params.verifier = verifier
    actions = actions or [MutationTypesEnum.single_change, MutationTypesEnum.single_add]
    algorithm = GPAlgorithmParameters(mutation_types=actions, mutation_prob=1.0,
                                      adaptive_mutation_type=RandomAgent(actions, [1 / len(actions)] * len(actions)))
    operator = PlannedMutation(Mutation(algorithm, requirements, params), params, policy or MutationPlanningPolicy())
    return operator, Individual(graph)


def _context():
    return SimpleNamespace(adapter=PipelineAdapter(), advisor=SimpleNamespace(task="regression"))


def _graph():
    feature = PrimaryNode("scaling")
    feature.parameters = {"with_mean": False}
    model = SecondaryNode("ridge", nodes_from=[feature])
    model.parameters = {"alpha": 10.0}
    return PipelineAdapter().adapt(Pipeline(model))


def test_runtime_projection_is_independent_of_fresh_node_uuids():
    first, second = _graph(), _graph()
    assert first.nodes[0].uid != second.nodes[0].uid
    assert project_for_mutation(first, _context()) == project_for_mutation(second, _context())


def test_interpreter_resets_replaced_parameters_and_does_not_alias_input():
    graph = _graph()
    before = project_for_mutation(graph, _context())
    root = next(node for node in before.nodes if node.operation_name == "ridge")
    after = replace(before, nodes=tuple(
        replace(node, operation_name="dtreg") if node.node_id == root.node_id else node for node in before.nodes))
    result = interpret_graph_spec(graph, before, after)
    by_name = {node.content["name"]: node for node in result.nodes}
    assert by_name["dtreg"].content["params"] == {}
    assert by_name["scaling"].content["params"] == {"with_mean": False}
    by_name["scaling"].content["params"]["with_mean"] = True
    assert graph.nodes[1].content["params"] == {"with_mean": False}
    assert graph.nodes[0].content["name"] == "ridge"


def test_interpreter_rejects_mismatched_projection():
    graph = _graph()
    before = project_for_mutation(graph, _context())
    with pytest.raises(ValueError, match="projection"):
        interpret_graph_spec(graph, replace(before, nodes=()), before)


def test_catalog_uses_factory_repository_and_keeps_position_restrictions():
    class Repository:
        def get_operations(self, is_primary):
            return ["scaling", "ridge"] if is_primary else ["ridge", "pca"]

    factory = SimpleNamespace(graph_model_repository=Repository(),
                              get_all_available_operations=lambda: pytest.fail("unimplemented accessor"))
    catalog = load_runtime_operation_catalog(SimpleNamespace(node_factory=factory))
    by_name = {operation.name: operation for operation in catalog.capabilities}
    assert set(by_name) == {"scaling", "ridge", "pca"}
    assert not by_name["pca"].supports_position(NodePosition.PRIMARY)
    assert by_name["pca"].supports_position(NodePosition.SECONDARY)
    assert by_name["ridge"].supports_position(NodePosition.ROOT)
    assert catalog.unavailable == ()


def test_catalog_with_no_declared_operation_pool_does_not_invent_models():
    catalog = load_runtime_operation_catalog(SimpleNamespace(node_factory=SimpleNamespace(
        get_all_available_operations=lambda: None)))
    assert catalog.capabilities == ()


def test_interpreter_retains_shared_branch_topology():
    graph = _graph()
    before = project_for_mutation(graph, _context())
    feature = next(n for n in before.nodes if n.operation_name == "scaling")
    root = next(n for n in before.nodes if n.operation_name == "ridge")
    extra = replace(feature, node_id="extra", operation_name="pca", parent_ids=(feature.node_id,))
    after = replace(before, nodes=(replace(root, parent_ids=(feature.node_id, "extra")), feature, extra))
    result = interpret_graph_spec(graph, before, after)
    root_node, feature_node, extra_node = result.nodes
    assert root_node.nodes_from == [feature_node, extra_node]
    assert extra_node.nodes_from[0] is feature_node


def test_advisor_veto_is_preserved_for_replacement_and_removal():
    graph = _graph()
    before = project_for_mutation(graph, _context())
    root = before.nodes[0]
    after = replace(before, nodes=(replace(root, operation_name="dtreg"), *before.nodes[1:]))
    advisor = SimpleNamespace(propose_change=lambda node, candidates: [])
    assert not advisor_allows_plan(before, after, graph, advisor)
    removed = replace(before, nodes=(root,))
    advisor = SimpleNamespace(can_be_removed=lambda node: "forbidden")
    assert not advisor_allows_plan(before, removed, graph, advisor)


@pytest.mark.parametrize("permission", ["with_parents", "with_direct_children"])
def test_bypass_does_not_ignore_advisor_subtree_removal_semantics(permission):
    graph = _graph()
    before = project_for_mutation(graph, _context())
    assert not advisor_allows_plan(before, replace(before, nodes=(before.nodes[0],)), graph,
                                   SimpleNamespace(can_be_removed=lambda node: permission))


def test_new_parent_respects_advisor_veto():
    graph = _graph()
    before = project_for_mutation(graph, _context())
    extra = replace(before.nodes[1], node_id="extra")
    after = replace(before, nodes=(replace(before.nodes[0], parent_ids=("extra",)), before.nodes[1], extra))
    assert not advisor_allows_plan(before, after, graph, SimpleNamespace(propose_parent=lambda node, candidates: []))


def test_runtime_plan_is_deterministic_replayable_and_does_not_mutate_parent():
    operator, parent = _planned()
    original = deepcopy(parent.graph)
    with operator_random_scope(123):
        first = operator(parent)
    with operator_random_scope(123):
        second = operator(parent)
    assert isinstance(first, Individual)
    assert first.graph == second.graph
    assert first.graph != parent.graph
    assert parent.graph == original
    assert first.metadata["mutation_plan"] == second.metadata["mutation_plan"]
    assert first.metadata["mutation_input"]["schema_version"] == "industrial_graph_spec@1"
    assert operator.last_execution.status is MutationExecutionStatus.ACCEPTED
    assert operator.last_execution.examined <= operator.policy.max_candidates


def test_no_admissible_action_does_not_fall_back_to_unconfigured_operations():
    operator, parent = _planned([MutationTypesEnum.single_edge])
    assert operator(parent) == []
    assert operator.last_execution.status is MutationExecutionStatus.NO_ADMISSIBLE_PLAN


def test_runtime_verifier_can_reject_a_structurally_valid_plan_once():
    calls = []
    operator, parent = _planned(verifier=lambda graph: calls.append(graph) or False)
    assert operator(parent) == []
    assert len(calls) == 1
    assert operator.last_execution.status is MutationExecutionStatus.REJECTED


def test_parameter_callback_executes_once_on_a_copy():
    calls = []

    def set_alpha(pipeline, **kwargs):
        calls.append(True)
        pipeline.root_node.parameters = {"alpha": 7.0}
        return pipeline

    operator, parent = _planned([set_alpha])
    child = operator(parent)
    assert isinstance(child, Individual)
    assert calls == [True]
    assert child.graph.nodes[0].content["params"] == {"alpha": 7.0}
    assert parent.graph.nodes[0].content["params"] != {"alpha": 7.0}
    assert "mutation_plan" not in child.metadata


def test_zero_probability_preserves_parent_and_empty_population_is_supported():
    operator, parent = _planned()
    operator.parameters.mutation_prob = 0.0
    assert operator(parent) is parent
    assert operator.last_execution.status is MutationExecutionStatus.SKIPPED
    assert operator([]) == []


def test_unknown_task_is_a_structured_rejection_before_random_choice():
    operator, parent = _planned()
    operator.graph_generation_params.advisor.task = None
    assert operator(parent) == []
    assert operator.last_execution.status is MutationExecutionStatus.REJECTED
    assert "explicit graph task" in operator.last_execution.message


def test_mutation_weights_align_with_action_names_and_normalize_subsets():
    strategy = [.15, .15, .30, .10, .30]
    assert mutation_weights([MutationTypesEnum.single_add, MutationTypesEnum.single_change], strategy) == [.4, .6]
    assert mutation_weights([MutationTypesEnum.single_drop], strategy) == [1.0]
    assert mutation_weights([], strategy) == []
    with pytest.raises(ValueError, match="five"):
        mutation_weights([MutationTypesEnum.single_add], [1.0])


def test_planned_operator_worker_state_does_not_capture_coordinator_callback():
    import cloudpickle
    operator, parent = _planned()
    operator.observer = lambda execution: pytest.fail("coordinator callback copied to worker")
    restored = cloudpickle.loads(cloudpickle.dumps(operator))
    assert restored.observer is None
    assert operator.observer is not None
    assert isinstance(restored(parent), Individual)
