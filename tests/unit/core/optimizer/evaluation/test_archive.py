"""Final selection covers completed evaluations, not just reproduction survivors."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import random

import numpy as np
import pytest
from hypothesis import given, strategies as st
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.fitness import MultiObjFitness, SingleObjFitness, null_fitness
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult, Individual

from fedot_ind.core.optimizer.domain import EvaluationSucceeded, evaluation_outcome_to_metadata
from fedot_ind.core.optimizer.evaluation.archive import (
    CompletedEvaluationArchive, final_history_choices, record_final_history,
)
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json


def _result(uid='candidate', score=.9, *, episode=None):
    graph = OptGraph(OptNode({'name': 'ridge', 'params': {'alpha': 1.}}))
    configuration = {'candidate': uid}
    identity = PreparedEvaluationGraph(graph, canonical_json(configuration)).configuration_id
    metadata = evaluation_outcome_to_metadata(EvaluationSucceeded(.1, 'frozen-time'))
    metadata['evaluation_context'] = {'candidate_id': uid, 'evaluation_id': episode or f'{uid}:0',
                                      'configuration_id': identity, 'configuration': configuration, 'seed': 42}
    return GraphEvalResult(uid, SingleObjFitness(score), graph, metadata)


def _archive(keep=1):
    return CompletedEvaluationArchive(Objective({'loss': lambda _: 0.}), keep_n_best=keep)


@given(st.lists(st.floats(min_value=-1.e6, max_value=1.e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=35))
def test_completed_minimum_is_monotonic_and_every_episode_is_retained(scores):
    archive = _archive()
    for index, score in enumerate(scores):
        archive.record_results([_result(str(index), score)])
        assert archive.best_individuals[0].fitness.value == min(scores[:index + 1])
        assert len(archive.entries) == index + 1


def test_snapshots_detach_graph_metadata_fitness_and_do_not_consume_rng():
    archive = _archive()
    result = _result()
    result.metadata['nested'] = {'values': [1]}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    archive.record_results([result])
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    result.graph.nodes[0].content['params']['alpha'] = 999
    result.metadata['nested']['values'].append(2)
    first = archive.best_individuals[0]
    assert first.graph.nodes[0].content['params']['alpha'] == 1.
    assert first.metadata['nested']['values'] == [1]
    assert first.parent_operator is None
    first.graph.nodes[0].content['params']['alpha'] = -1
    first.metadata['nested']['values'].clear()
    assert archive.best_individuals[0].metadata['nested']['values'] == [1]
    assert archive.best_individuals[0].graph.nodes[0].content['params']['alpha'] == 1.
    with pytest.raises(FrozenInstanceError):
        archive.entries[0].candidate_id = 'changed'


def test_episode_replay_is_idempotent_conflicts_fail_and_repeated_uid_can_improve():
    archive = _archive()
    first = _result()
    archive.record_results([first, deepcopy(first)])
    assert len(archive.entries) == 1
    conflicting = deepcopy(first)
    conflicting.fitness = SingleObjFitness(.8)
    with pytest.raises(ValueError, match='conflicting'):
        archive.record_results([conflicting])
    archive.record_results([_result(score=.1, episode='candidate:1')])
    assert len(archive.entries) == 2
    assert archive.best_individuals[0].fitness.value == .1
    assert archive.best_individuals[0].metadata['evaluation_context']['evaluation_id'] == 'candidate:1'


@pytest.mark.parametrize('damage', ['null', 'nan', 'inf', 'failed', 'reused', 'partial', 'none'])
def test_failed_partial_reused_and_nonfinite_results_are_not_archived(damage):
    archive = _archive()
    result = _result()
    if damage == 'null':
        result.fitness = null_fitness()
    elif damage in ('nan', 'inf'):
        result.fitness = SingleObjFitness(float(damage))
    elif damage == 'partial':
        result.metadata['evaluation_failure'] = {'code': 'objective_incomplete', 'message': 'missing fold'}
    elif damage == 'none':
        result = None
    else:
        result.metadata['evaluation_status'] = damage
    archive.record_results([result])
    assert archive.entries == () and archive.best_individuals == ()


def test_context_identity_cannot_be_reassigned_to_another_candidate():
    archive = _archive()
    result = _result()
    result.metadata['evaluation_context']['candidate_id'] = 'other'
    with pytest.raises(ValueError, match='another candidate'):
        archive.record_results([result])
    assert not archive.entries


def test_native_ties_keep_first_and_keep_n_best_uses_native_fitness_order():
    archive = _archive(keep=2)
    archive.record_results([_result('first', .1), _result('second', .1), _result('third', 1.)])
    assert archive.best_individuals[0].uid in ('first', 'second')
    assert {item.uid for item in archive.best_individuals} == {'first', 'second'}
    single = _archive()
    single.record_results([_result('first', .1), _result('second', .1)])
    assert single.best_individuals[0].uid == 'first'


def test_multiobjective_front_preserves_nondominated_candidates_and_excludes_nonfinite():
    archive = CompletedEvaluationArchive(Objective({'a': lambda _: 0., 'b': lambda _: 0.}, is_multi_objective=True))
    rows = [_result(str(i)) for i in range(4)]
    for result, values in zip(rows, ((1., 2.), (2., 1.), (3., 3.), (0., float('nan')))):
        result.fitness = MultiObjFitness(values, weights=(1., 1.))
    archive.record_results(rows)
    assert len(archive.entries) == 3
    assert {item.fitness.values for item in archive.best_individuals} == {(1., 2.), (2., 1.)}


def test_weighted_multiobjective_values_are_preserved_without_second_weighting():
    archive = CompletedEvaluationArchive(Objective({'a': lambda _: 0., 'b': lambda _: 0.}, is_multi_objective=True))
    first, second = _result('first'), _result('second')
    first.fitness = MultiObjFitness((1., 5.), weights=(1., -1.))
    second.fitness = MultiObjFitness((2., 1.), weights=(1., -1.))
    archive.record_results([first, second])
    selected, = archive.best_individuals
    assert selected.uid == 'first'
    assert selected.fitness.values == (1., -5.)
    assert selected.fitness.weights == (1., -1.)
    assert archive.entries[0].fitness == (1., -5.)


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
def test_dispatcher_archives_before_consumer_truncation_and_does_not_duplicate_reuse(backend):
    archive = _archive()
    dispatcher = LocalEvaluationDispatcher(DirectAdapter(), n_jobs=1,
                                           config=EvaluationExecutionConfig(backend),
                                           completed_evaluation_callback=archive.record_results)
    individuals = [Individual(OptGraph(OptNode(str(i)))) for i in range(7)]

    def objective(graph):
        return SingleObjFitness(.85 if str(graph.nodes[0]) == '6' else .9)

    evaluate = dispatcher.dispatch(objective)
    evaluated = evaluate(individuals)
    assert len(archive.entries) == 7
    assert archive.best_individuals[0].uid == individuals[-1].uid
    generation = GenerationKeeper(Objective({'loss': lambda _: 0.}))
    generation.append(evaluated[:4])
    assert generation.best_individuals[0].fitness.value == .9
    assert archive.best_individuals[0].fitness.value == .85
    evaluate(individuals)
    assert len(archive.entries) == 7


def test_cleanup_failure_cannot_enter_completed_archive():
    archive = _archive()

    def fail_cleanup(graph):
        raise RuntimeError('cleanup failed')

    dispatcher = LocalEvaluationDispatcher(DirectAdapter(),
                                           config=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL),
                                           graph_cleanup_fn=fail_cleanup,
                                           completed_evaluation_callback=archive.record_results)
    assert not dispatcher.dispatch(lambda _: SingleObjFitness(1.))([Individual(OptGraph(OptNode('ridge')))])
    assert not archive.entries


def test_final_minimum_is_exact_even_when_native_fitness_considers_values_close():
    archive = _archive()
    archive.record_results([_result('earlier', 1.), _result('later', 1. - 1.e-11)])
    assert SingleObjFitness(1.) == SingleObjFitness(1. - 1.e-11)
    assert archive.best_individuals[0].uid == 'later'
    assert archive.best_individuals[0].fitness.value == 1. - 1.e-11


@pytest.mark.parametrize('keep_history', [False, True])
def test_public_optimizer_returns_discarded_minimum_without_changing_generations(tmp_path, keep_history):
    from datetime import timedelta
    import dask
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy

    requirements, params, _ = _environment(
        {'name': 'archive-test', 'task': 'regression', 'operations': ['ridge', 'dtreg'],
         'initial_chain': ['ridge']}, {'max_depth': 4, 'max_arity': 2})
    requirements.num_of_generations = 1
    requirements.timeout = timedelta(seconds=120)
    requirements.keep_history = keep_history
    requirements.agent_dir = str(tmp_path)

    def graph(score):
        return params.adapter.adapt(PipelineBuilder().add_node('ridge', params={'alpha': score}).build())

    def score(pipeline):
        return pipeline.root_node.parameters['alpha']

    objective = Objective({'loss': score})
    optimizer = IndustrialEvoOptimizer(
        objective, [graph(value) for value in (1., 1.1, 1.2, 1.3)], requirements, params,
        GPAlgorithmParameters(pop_size=4, max_pop_size=4),
        EvolutionConfig(random_seed=42, mutation_planning=MutationPlanningPolicy(enabled=False),
                        evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL)))
    offspring = [Individual(graph(value)) for value in (.9, 1., 1.1, 1.2, 1.3, 1.4, .85)]
    generation_choices = []

    def reproduce(population, evaluate):
        evaluated = evaluate(offspring)
        assert len(optimizer.completed_evaluations.entries) == 11
        assert optimizer.completed_evaluations.best_individuals[0].fitness.value == .85
        assert optimizer.best_individuals[0].fitness.value == 1.
        return evaluated[:4]

    optimizer.reproducer.reproduce = reproduce
    optimizer.mutation.agent.partial_fit = lambda experience: None  # Reproduction is fixed, not agent-driven.
    optimizer.set_iteration_callback(lambda population, owner: generation_choices.append(
        (bool(owner.final_individuals), tuple(item.fitness.value for item in population))))
    with dask.config.set(scheduler='synchronous'):
        result = optimizer.optimise(objective)
    assert params.adapter.restore(result[0]).root_node.parameters['alpha'] == .85
    assert optimizer.best_individuals[0].fitness.value == .85
    assert optimizer.generations.best_individuals[0].fitness.value == .9
    assert optimizer.current_generation_num == 2
    assert len(optimizer.evaluated_population) == 2
    assert .85 not in generation_choices[1][1]
    assert generation_choices[-1] == (True, (.85,))
    if keep_history:
        from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
        restored = OptHistory.load(optimizer.history.save())
        assert restored.generations[-1].label == 'final_choices'
        assert restored.generations[-1][0].uid == offspring[-1].uid
        assert restored.generations[-1][0].native_generation is None


def test_final_history_round_trip_preserves_uid_owner_metadata_and_parent_lineage():
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
    parent = Individual(OptGraph(OptNode('ridge')), metadata={'origin': 'initial'}, uid='parent',
                        fitness=SingleObjFitness(1.))
    result = _result('child', .1)
    child = Individual(result.graph, metadata={**deepcopy(result.metadata), 'origin': 'mutation'}, uid='child',
                       fitness=result.fitness,
                       parent_operator=ParentOperator('mutation', 'single_change', (parent,)))
    history = OptHistory()
    history.add_to_history([parent], 'initial')
    history.add_to_archive_history([parent])
    history.add_to_history([child], 'evolved')
    history.add_to_archive_history([child])
    original = OptHistory.load(history.save())
    archive = _archive()
    archive.record_results([result])
    final = final_history_choices(history.generations, archive.best_individuals)
    assert final[0] is child  # Keep the existing UID owner, not a stripped archive snapshot.
    record_final_history(history, archive.best_individuals)
    restored = OptHistory.load(history.save())
    for before, after in zip(original.generations, restored.generations):
        assert [item.uid for item in before] == [item.uid for item in after]
        for left, right in zip(before, after):
            assert left.metadata == right.metadata
            assert left.native_generation == right.native_generation
            assert [item.uid for item in left.parents] == [item.uid for item in right.parents]
    assert restored.generations[-1][0].uid == 'child'
    assert restored.generations[-1][0].parents[0].uid == 'parent'


def test_history_choices_find_intermediate_parents_and_reject_conflicting_uid():
    from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
    parent = Individual(OptGraph(OptNode('ridge')), uid='parent', fitness=SingleObjFitness(.1))
    child = Individual(OptGraph(OptNode('ridge')), uid='child', fitness=SingleObjFitness(.2),
                       parent_operator=ParentOperator('mutation', 'single_change', (parent,)))
    snapshot = Individual(deepcopy(parent.graph), uid=parent.uid, fitness=deepcopy(parent.fitness))
    assert final_history_choices([[child]], [snapshot]) == (parent,)
    snapshot.fitness.values = (.3,)
    with pytest.raises(ValueError, match='historical UID'):
        final_history_choices([[child]], [snapshot])


def test_final_intermediate_parent_keeps_missing_native_generation_after_round_trip():
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
    from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
    result = _result('intermediate', .1)
    parent = Individual(result.graph, uid='intermediate', metadata=deepcopy(result.metadata), fitness=result.fitness)
    child = Individual(OptGraph(OptNode('ridge')), uid='child', fitness=SingleObjFitness(.2),
                       parent_operator=ParentOperator('mutation', 'single_change', (parent,)))
    history = OptHistory()
    history.add_to_history([child], 'evolved')
    history.add_to_archive_history([child])
    archive = _archive()
    archive.record_results([result])
    record_final_history(history, archive.best_individuals)
    assert parent.native_generation is None
    restored = OptHistory.load(history.save())
    assert restored.generations[-1].label == 'final_choices'
    assert restored.generations[-1][0].native_generation is None
    assert restored.generations[0][0].parents[0].native_generation is None


def test_legacy_coordinator_records_fresh_completions_but_worker_does_not_own_archive():
    from types import SimpleNamespace
    from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
    archive = _archive()
    dispatcher = object.__new__(IndustrialDispatcher)
    dispatcher._adapter = SimpleNamespace(adapt_func=lambda function: function)
    dispatcher.logger = SimpleNamespace(info=lambda message: None)
    dispatcher.diagnostics_recorder = None
    dispatcher.completed_evaluation_callback = archive.record_results
    dispatcher._evaluate_graph = lambda graph: (SingleObjFitness(.5), graph)
    result = dispatcher.eval_ind(OptGraph(OptNode('ridge')), 'candidate').compute(scheduler='synchronous')
    assert archive.entries == ()
    dispatcher._record_evaluation_results([result])
    assert len(archive.entries) == 1
    assert dispatcher.__getstate__()['completed_evaluation_callback'] is None
    dispatcher._record_reused_individuals([Individual(result.graph, fitness=result.fitness)])
    assert len(archive.entries) == 1
