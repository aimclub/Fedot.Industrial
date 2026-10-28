"""TRAIN-only boundary and evaluation-to-generation selection evidence."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from golem.core.optimisers.fitness import SingleObjFitness

from benchmark.industrial.experiments.mutation_planning.population_study.preparation import (
    selected_evaluation, selection_summary, completed_objective_calls,
)
from benchmark.industrial.experiments.mutation_planning.population_study.protocol import load_protocol
from benchmark.industrial.experiments.mutation_planning.population_study.selection_recheck.data import (
    load_training_data, training_data_guard,
)
from benchmark.industrial.experiments.mutation_planning.population_study.selection_recheck.runner import load_settings
from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json


def _selected():
    prepared = PreparedEvaluationGraph('graph', canonical_json({'nodes': ['frozen']}))
    context = {'candidate_id': 'candidate', 'evaluation_id': 'candidate:0',
               'configuration_id': prepared.configuration_id, 'configuration': {'nodes': ['frozen']}, 'seed': 42}
    individual = SimpleNamespace(uid='candidate', metadata={'evaluation_context': context},
                                 fitness=SingleObjFitness(.9))
    call = {**deepcopy(context), 'valid': True, 'fitness': .9, 'fold_count': 1}
    return individual, prepared, call


def test_selected_evidence_is_an_exact_detached_episode():
    individual, prepared, call = _selected()
    actual = selected_evaluation(individual, prepared, [call])
    assert actual == individual.metadata['evaluation_context']
    actual['configuration']['nodes'].append('changed')
    assert call['configuration']['nodes'] == ['frozen']
    assert individual.metadata['evaluation_context']['configuration']['nodes'] == ['frozen']


@pytest.mark.parametrize('damage', ['uid', 'configuration', 'id', 'fitness', 'seed', 'invalid', 'folds',
                                    'missing', 'duplicate'])
def test_wrong_or_aliased_selected_episode_cannot_pass(damage):
    individual, prepared, call = _selected()
    calls = [call]
    if damage == 'uid':
        individual.uid = 'other-candidate-with-identical-configuration'
    elif damage == 'configuration':
        individual.metadata['evaluation_context']['configuration']['nodes'].append('changed')
    elif damage == 'id':
        individual.metadata['evaluation_context']['configuration_id'] = 'other'
    elif damage == 'fitness':
        call['fitness'] = .8
    elif damage == 'seed':
        call['seed'] = 123
    elif damage == 'invalid':
        call['valid'] = False
    elif damage == 'folds':
        call['fold_count'] = 0
    elif damage == 'missing':
        calls = []
    else:
        calls.append(deepcopy(call))
    with pytest.raises(ValueError):
        selected_evaluation(individual, prepared, calls)


def test_seventh_valid_evaluation_can_be_outside_four_admitted_candidates():
    scores = [.9, 1., 1.1, 1.2, 1.3, 1.4, .85]
    calls = [{'valid': True, 'candidate_id': str(i), 'evaluation_id': f'{i}:0', 'fitness': score}
             for i, score in enumerate(scores)]
    members = [{'evaluation_id': f'{i}:0', 'candidate_id': str(i), 'fitness': scores[i]} for i in range(4)]
    selected = {'fitness': .9, 'evaluation_context': {'evaluation_id': '0:0'}}
    report = selection_summary(selected, calls, [{'members': members}])
    assert report['observed_minimum'] == .85
    assert report['generation_population_minimum'] == .9
    assert not report['selected_is_observed_minimum']
    assert report['selected_is_generation_minimum']
    assert [row['candidate_id'] for row in report['outside_generation_population']] == ['4', '5', '6']
    assert report == selection_summary(selected, list(reversed(calls)), [{'members': list(reversed(members))}])


def test_unobserved_generation_episode_is_rejected():
    with pytest.raises(ValueError, match='successful evaluation'):
        selection_summary({'fitness': 1}, [], [{'members': [{'evaluation_id': 'unknown'}]}])


def test_completed_winner_outside_generations_is_not_relabelled_as_a_generation_member():
    calls = [{'valid': True, 'candidate_id': str(i), 'evaluation_id': f'{i}:0', 'fitness': score}
             for i, score in enumerate([.9, .85])]
    generations = [{'members': [{'candidate_id': '0', 'evaluation_id': '0:0', 'fitness': .9}]},
                   {'is_final': True, 'members': [{'candidate_id': '1', 'evaluation_id': '1:0', 'fitness': .85}]}]
    selected = {'fitness': .85, 'evaluation_context': {'evaluation_id': '1:0'}}
    report = selection_summary(selected, calls, generations)
    assert report['selected_is_observed_minimum']
    assert not report['selected_is_generation_minimum']
    assert not report['selected_entered_generation_population']
    assert report['generation_population_minimum'] == .9


@pytest.mark.parametrize('name', ['Daily-test.csv', 'Example_TEST.ts', 'example_test.tsv',
                                  'result.json', 'records.json'])
def test_guard_blocks_test_and_mixed_results_before_open(tmp_path, name):
    # The nonexistent path proves the guard rejects before the filesystem is read.
    config = {'forecast_test': str(tmp_path / 'Daily-test.csv')}
    with training_data_guard(config, {'name': 'D1'}) as accesses:
        with pytest.raises(RuntimeError, match='TRAIN-only'):
            (tmp_path / name).read_bytes()
    assert len(accesses['blocked']) == 1


def test_guard_tracks_training_reads_and_disables_itself_after_exit(tmp_path):
    train = tmp_path / 'Daily-train.csv'
    train.write_text('D1,1,2,3\n', encoding='utf-8')
    forbidden = tmp_path / 'any-name.csv'
    forbidden.write_text('not-read-within-guard', encoding='utf-8')
    with training_data_guard({'forecast_test': str(forbidden)}, {}) as accesses:
        assert train.read_text(encoding='utf-8').startswith('D1')
        with pytest.raises(RuntimeError):
            forbidden.read_bytes()
    assert accesses['opened_training_files'] == [str(train.resolve())]
    assert forbidden.read_text(encoding='utf-8') == 'not-read-within-guard'


def test_forecast_train_loader_has_no_test_dependency_and_preserves_inner_split(tmp_path):
    path = tmp_path / 'Daily-train.csv'
    history = np.arange(40, dtype=float)
    path.write_text('D1,' + ','.join(map(str, history)) + '\n', encoding='utf-8')
    config = {'forecast_train': str(path), 'forecast_test': str(tmp_path / 'DOES_NOT_EXIST.csv')}
    with training_data_guard(config, {}) as accesses:
        data = load_training_data(config, {'name': 'D1', 'task': 'ts_forecasting'}, 42, forecast_horizon=14)
    np.testing.assert_array_equal(data.train.features.numpy().reshape(-1), history[:-14])
    np.testing.assert_array_equal(data.validation.target.numpy().reshape(-1), history[-14:])
    assert data.scale == 1.
    assert set(data.source['files']) == {str(path)}
    assert not accesses['blocked'] and accesses['opened_training_files']
    assert not hasattr(data, 'test')
    with pytest.raises(ValueError, match='explicit frozen'):
        load_training_data(config, {'name': 'D1', 'task': 'ts_forecasting'}, 42, forecast_horizon=True)


def test_regression_inner_indices_dtype_and_scale_match_original_protocol(tmp_path, monkeypatch):
    from sklearn.model_selection import train_test_split
    from benchmark.industrial.experiments.mutation_planning.population_study import supervised_diagnostics
    x = np.arange(60, dtype=float).reshape(12, 5)
    y = np.sin(np.arange(12)).astype(float)
    path = tmp_path / 'Fixture_TRAIN.ts'
    path.write_text('fixture', encoding='utf-8')
    monkeypatch.setattr(supervised_diagnostics, '_train_arrays', lambda *_: (x, y, path))
    data = load_training_data({}, {'name': 'Fixture', 'task': 'regression'}, 42)
    tr, val = train_test_split(np.arange(12), test_size=.25, random_state=42)
    np.testing.assert_array_equal(data.train.idx, tr)
    np.testing.assert_array_equal(data.validation.idx, val)
    np.testing.assert_array_equal(data.train.features.numpy()[:, 0], x[tr].astype(np.float32))
    np.testing.assert_array_equal(data.validation.target.numpy(), y[val])
    assert str(data.train.target.dtype) == 'torch.float64'
    assert data.scale == float(np.std(y[tr]))


def test_settings_keep_four_searches_and_do_not_replace_the_full_protocol():
    settings = load_settings()
    assert len(settings['jobs']) == 4
    assert settings['forecast_horizon'] == 14
    protocol = load_protocol(
        'benchmark/industrial/experiments/mutation_planning/population_study/protocol_20261019.json')
    assert protocol.digest == 'a524f37977ba673a0e35e0e9166de508f38d43206e5ae9ad9478ec90c311e455'
    assert json.loads(json.dumps(settings)) == settings


def test_completed_archive_recheck_contains_only_the_two_approved_bitcoin_searches():
    settings = load_settings('benchmark/industrial/experiments/mutation_planning/population_study/'
                             'selection_recheck/archive_defaults.json')
    assert settings['jobs'] == [{'dataset': 'BitcoinSentiment', 'seed': 42, 'mode': mode}
                                for mode in ('legacy', 'planned')]


def test_valid_objective_return_with_failed_completion_cannot_win_selection():
    individual, prepared, call = _selected()
    rejected = {**deepcopy(call), 'evaluation_id': 'timed-out:1', 'candidate_id': 'timed-out', 'fitness': .1}
    entries = [{'evaluation_id': call['evaluation_id'], 'candidate_id': call['candidate_id'],
                'fitness': [call['fitness']],
                'evaluation_context': deepcopy(individual.metadata['evaluation_context'])}]
    calls = [call, rejected]
    assert completed_objective_calls(calls, entries) == [call]
    generations = [{'members': [{'evaluation_id': call['evaluation_id'],
                                 'candidate_id': call['candidate_id'], 'fitness': call['fitness']}]}]
    report = selection_summary({'fitness': .9, 'evaluation_context': individual.metadata['evaluation_context']},
                               calls, generations, completed_entries=entries)
    assert report['observed_minimum'] == .9 and report['objective_return_minimum'] == .1
    assert report['completed_evaluation_count'] == 1


def test_native_initial_dispatch_keeps_different_forecast_configurations_attached():
    from fedot.core.pipelines.adapters import PipelineAdapter
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
    from fedot_ind.core.optimizer.evaluation.context import current_evaluation_context
    from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import StudyGraphPreparation

    adapter = PipelineAdapter()
    parameters = load_protocol().to_record()['operation_parameters']
    preparation = StudyGraphPreparation.from_parameters(adapter, parameters, 'ts_forecasting')
    calls = []

    def objective(pipeline):
        context = current_evaluation_context().to_record()
        score = .1 if pipeline.root_node.operation.operation_type == 'mssa_forecaster' else .9
        calls.append({**context, 'valid': True, 'fold_count': 1, 'fitness': score})
        return SingleObjFitness(score)

    with industrial_extension_scope():
        graphs = [adapter.adapt(PipelineBuilder().add_node('mssa_forecaster').build()),
                  adapter.adapt(PipelineBuilder().add_node('hankelisation').add_node('ridge').build())]
        optimizer = object.__new__(IndustrialEvoOptimizer)
        optimizer.requirements = SimpleNamespace(static_individual_metadata={})
        optimizer.graph_optimizer_params = SimpleNamespace(pop_size=2)
        optimizer._admissible_initial_graphs = lambda: graphs
        optimizer._update_population = lambda **kwargs: None
        dispatcher = LocalEvaluationDispatcher(adapter, config=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL),
                                               random_seed=42, graph_preparation=preparation)
        population, _ = optimizer._initial_population(dispatcher.dispatch(objective))
        winner = min(population, key=lambda individual: individual.fitness.value)
        evidence = selected_evaluation(winner, preparation(winner.graph), calls)
    assert evidence['candidate_id'] == winner.uid == calls[0]['candidate_id']
    assert population[0].metadata['evaluation_context'] != population[1].metadata['evaluation_context']


def test_native_reproduction_truncation_explains_valid_but_unadmitted_minimum():
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from golem.core.optimisers.graph import OptGraph, OptNode
    from golem.core.optimisers.archive.generation_keeper import GenerationKeeper
    from golem.core.optimisers.objective import Objective

    parameters = GPAlgorithmParameters(pop_size=4, required_valid_ratio=1.)
    controller = ReproductionController(parameters, None, None, None)
    candidates = [Individual(OptGraph(OptNode(str(i))), uid=str(i), fitness=SingleObjFitness(score))
                  for i, score in enumerate([.9, 1., 1.1, 1.2, 1.3, 1.4, .85])]
    batches = iter([candidates[:3], candidates[3:]])
    published = []

    def evaluate_batch(*args):
        batch = next(batches)
        published.extend(batch)
        return batch

    controller.reproduce_uncontrolled = evaluate_batch
    admitted = controller.reproduce(candidates[:4], None)
    keeper = GenerationKeeper(Objective({'loss': lambda _: 0.}))
    keeper.append(admitted)
    assert [candidate.uid for candidate in published] == list(map(str, range(7)))
    assert [candidate.uid for candidate in admitted] == ['0', '1', '2', '3']
    assert min(candidate.fitness.value for candidate in published) == .85
    assert keeper.best_individuals[0].fitness.value == .9
