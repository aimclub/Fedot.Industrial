from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from benchmark.industrial.experiments.mutation_planning.runner import _environment
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy
from fedot_ind.core.optimizer.search_policy import MutationMix, parse_staged_search
from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
from fedot_ind.core.optimizer.randomness import operator_random_scope
from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
from fedot_ind.integration.fedot.evaluation import ConfirmedObjective, confirmed_input_from_objective


def setup_optimizer(weights=None, task='regression'):
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.objective import Objective
    import fedot_ind.core.optimizer.search_policy as module
    policy = parse_staged_search(json.loads(Path(module.__file__).with_name('defaults.json').read_text('utf-8')))
    if weights is not None:
        policy = replace(policy, stages=tuple(replace(s, weights=weights) for s in policy.stages))
    scenario = dict(name='staged', task='regression', operations=['scaling', 'normalization', 'ridge'],
                    initial_chain=['scaling', 'ridge'])
    if task == 'ts_forecasting':
        scenario = dict(name='staged_forecast', task=task, operations=['hankelisation', 'gaussian_filter', 'ridge'],
                        initial_chain=['hankelisation', 'ridge'])
    requirements, params, graph = _environment(scenario, dict(max_depth=6, max_arity=3))
    requirements.timeout = timedelta(seconds=100)
    parameters = GPAlgorithmParameters(mutation_prob=1., mutation_types=[MutationTypesEnum.single_change,
                                       MutationTypesEnum.single_add, MutationTypesEnum.single_drop])
    optimizer = IndustrialEvoOptimizer(Objective({'loss': lambda g: 0.}), [graph], requirements, params, parameters,
                                       EvolutionConfig(mutation_planning=MutationPlanningPolicy(enabled=True), mutation_schedule=policy))
    optimizer.mutation.raw_input = RawInputSpec('ts' if task == 'ts_forecasting' else 'tabular', True)
    return optimizer, graph, parameters


def test_forecasting_rewrite_keeps_the_mandatory_hankel_boundary():
    from fedot.core.operations.operation import Operation
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    with industrial_extension_scope():
        optimizer, graph, _ = setup_optimizer(MutationMix(0., 0., 0., 0., 1.), 'ts_forecasting')
        with patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')):
            with operator_random_scope(42):
                child = optimizer.mutation._mutate(Individual(graph))
        assert child is not None
        names = [node.content['name'] for node in child.graph.nodes]
        assert 'hankelisation' in names and 'gaussian_filter' in names
        assert optimizer.graph_generation_params.verifier(child.graph)


@pytest.mark.parametrize('weights,action', [(MutationMix(0., 0., 0., 0., 1.), 'rewrite_subgraph_mutation'),
                                            (MutationMix(.99, .01, 0., 0., 0.), 'parameter_change_mutation')])
def test_public_optimizer_wires_real_operators_without_training(weights, action):
    from fedot.core.operations.operation import Operation
    from golem.core.optimisers.opt_history_objects.individual import Individual
    optimizer, graph, original_parameters = setup_optimizer(weights)
    assert len(original_parameters.mutation_types) == 3
    if action == 'parameter_change_mutation':
        from fedot_ind.core.optimizer.mutation import mutation_name
        optimizer.mutation.agent._probs = [float(mutation_name(item) == action)
                                           for item in optimizer.mutation.agent.available_actions]
    snapshot = graph.descriptive_id
    with patch.object(Operation, 'fit', side_effect=AssertionError('Training forbidden')) as fit:
        with operator_random_scope(42):
            child = optimizer.mutation._mutate(Individual(graph))
        assert fit.call_count == 0
    assert child is not None and child.graph != graph
    assert child.metadata['mutation_execution']['action'] == action
    assert optimizer.graph_generation_params.verifier(child.graph)
    assert graph.descriptive_id == snapshot


def test_policy_changes_at_time_boundaries_and_retains_family_minima():
    optimizer, graph, _ = setup_optimizer()
    controller = optimizer.staged_search
    for seconds in (0., 35., 80.):
        controller.apply(optimizer.mutation, seconds)
    assert [r['name'] for r in controller.events] == ['explore', 'joint', 'refine']
    assert optimizer.mutation.agent._probs[3] == pytest.approx(.8)
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.individual import Individual
    a = Individual(graph, fitness=SingleObjFitness(2.))
    b = Individual(graph, fitness=SingleObjFitness(1.))
    controller.remember(a)
    controller.remember(b)
    assert len(controller.families) == 1
    assert controller.survivors([], 2)[0].fitness.value == 1.


def test_stage_trace_is_exported_without_mutating_the_old_diagnostics_contract(tmp_path):
    optimizer, _, _ = setup_optimizer()
    optimizer.staged_search.apply(optimizer.mutation, 0.)
    optimizer.staged_search.apply(optimizer.mutation, 80.)
    artifacts = optimizer.export_diagnostics(tmp_path)
    assert artifacts.events_jsonl.is_file()
    trace = json.loads((tmp_path / 'mutation_schedule.json').read_text('utf-8'))
    assert [event['name'] for event in trace['events']] == ['explore', 'refine']
    with pytest.raises(ValueError, match='backwards'):
        optimizer.staged_search.apply(optimizer.mutation, 10.)


def test_preparation_preserves_grid_values_and_canonicalizes_context():
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.core.optimizer.search_policy import ParameterGrid
    adapter = PipelineAdapter()
    pipeline = PipelineBuilder().add_node('ridge', params={'alpha': 10.}).build()
    grid = (ParameterGrid('ridge', 'alpha', (.1, 1., 10.)),)
    preparation = ParameterSearchPreparation.from_defaults(adapter, {'ridge': {'alpha': 1.}}, grid, 'regression')
    result = preparation(adapter.adapt(pipeline))
    assert json.loads(result.configuration_json)['nodes'][0]['parameters']['alpha'] == 10.
    assert preparation(result.graph).configuration_id == result.configuration_id
    pipeline.root_node.parameters = {'alpha': -1.}
    with pytest.raises(ValueError, match='outside'):
        preparation(adapter.adapt(pipeline))


def test_confirmed_wrapper_does_not_consume_a_fold_producer_and_propagates_failures():
    from fedot import create_data
    data = create_data(np.ones((6, 2)), target=np.arange(6), task='regression')
    calls = []
    wrapper = ConfirmedObjective.from_training_data(lambda graph: calls.append(graph), (data,))
    assert confirmed_input_from_objective(wrapper) == RawInputSpec('tabular', True)
    wrapper('graph')
    assert calls == ['graph']
    assert confirmed_input_from_objective(lambda graph: None) is None
    with pytest.raises(ValueError):
        ConfirmedObjective.from_training_data(wrapper, iter((data,)))


def test_optimizer_restores_confirmed_context_on_failure():
    from fedot import create_data
    from fedot_ind.core.optimizer.graph_validation import ModelFeatureFlow
    optimizer, _, _ = setup_optimizer()
    data = create_data(np.ones((6, 2)), target=np.arange(6), task='regression')
    confirmed = ConfirmedObjective.from_training_data(lambda graph: None, (data,))
    verifier = optimizer.graph_generation_params.verifier
    previous = (verifier.input_data_type, verifier.model_feature_flow, optimizer.mutation.raw_input)

    def fail(objective):
        assert verifier.input_data_type == 'tabular'
        assert verifier.model_feature_flow is ModelFeatureFlow.TENSOR_FEATURES
        assert optimizer.mutation.raw_input == confirmed.raw_input
        raise RuntimeError('expected failure')

    optimizer._optimise = fail
    with pytest.raises(RuntimeError, match='expected failure'):
        optimizer.optimise(confirmed)
    assert (verifier.input_data_type, verifier.model_feature_flow, optimizer.mutation.raw_input) == previous


def test_task_type_default_and_explicit_context_have_one_prepared_identity():
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from fedot.core.pipelines.adapters import PipelineAdapter
    with industrial_extension_scope():
        adapter = PipelineAdapter()
        preparation = ParameterSearchPreparation.from_defaults(adapter, {}, (), 'regression')
        before = PipelineBuilder().add_node('wavelet_basis', params={'wavelet': 'mexh'}).add_node('ridge').build()
        explicit = PipelineBuilder().add_node('wavelet_basis', params={
            'wavelet': 'mexh', 'task_type': 'regression'}).add_node('ridge').build()
        assert preparation(adapter.adapt(before)).configuration_id == preparation(
            adapter.adapt(explicit)).configuration_id
