from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import pytest
import fedot
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.extensions import get_registered_extensions
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.fitness import MultiObjFitness, SingleObjFitness, null_fitness
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult, Individual

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder


def _size(graph):
    assert get_registered_extensions()
    return SingleObjFitness(len(graph.nodes))


def _fail(graph):
    raise ValueError('bad objective')


def _individual():
    return Individual(OptGraph([OptNode('ridge')]))


def _dispatcher(backend=EvaluationBackend.SEQUENTIAL, **kwargs):
    return LocalEvaluationDispatcher(DirectAdapter(), n_jobs=2, config=EvaluationExecutionConfig(backend),
                                     diagnostics_recorder=EvolutionDiagnosticsRecorder(), random_seed=42, **kwargs)


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
def test_local_dispatcher_records_once_per_uid_and_preserves_registry(backend):
    before = tuple(get_registered_extensions())
    dispatcher = _dispatcher(backend)
    a, b = _individual(), _individual()
    evaluate = dispatcher.dispatch(_size)
    result = evaluate([a, a, b])
    assert [ind.uid for ind in result] == [a.uid, b.uid]
    assert dispatcher.diagnostics_recorder.snapshot().summary.evaluations_succeeded == 2
    assert tuple(get_registered_extensions()) == before
    assert all(ind.fitness.value == 1 for ind in result)
    assert evaluate([a, a]) == [a]
    assert dispatcher.diagnostics_recorder.snapshot().summary.evaluations_reused == 1
    assert dispatcher.last_plan.tasks == ()


def test_failure_cleans_graph_and_never_retries():
    cleaned = []
    dispatcher = _dispatcher(graph_cleanup_fn=lambda graph: cleaned.append(graph))
    result = dispatcher.dispatch(_fail)([_individual()])
    assert result == []
    assert len(cleaned) == 1
    assert dispatcher.diagnostics_recorder.snapshot().summary.evaluations_failed == 1
    assert not dispatcher.allows_untimed_retry
    assert dispatcher.evaluation_cache == {}


def test_cleanup_failure_is_not_a_success_and_cache_is_released():
    def cleanup(graph):
        raise RuntimeError('cleanup failed')

    dispatcher = _dispatcher(graph_cleanup_fn=cleanup)
    assert dispatcher.dispatch(_size)([_individual()]) == []
    failure, = dispatcher.last_results
    assert failure.error.message == 'cleanup failed'
    assert dispatcher.evaluation_cache == {}


def test_expired_timer_and_cancel_do_not_evaluate_objective():
    dispatcher = _dispatcher()
    dispatcher.dispatch(_size, SimpleNamespace(timeout=timedelta(seconds=1), seconds_from_start=2))
    assert dispatcher.evaluate_population([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'EvaluationTimeout'
    dispatcher.cancel()
    assert dispatcher.evaluate_population([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'Cancelled'
    dispatcher.dispatch(_size)
    dispatcher.cancel()
    assert dispatcher.evaluate_population([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'Cancelled'


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
@pytest.mark.parametrize('values', [None, (float('nan'),), (float('inf'),), (float('-inf'),), (0., float('nan'))])
def test_invalid_fitness_is_not_in_successful_population(backend, values):
    dispatcher = _dispatcher(backend)

    def objective(graph):
        if values is None:
            return null_fitness()
        return SingleObjFitness(values[0]) if len(values) == 1 else MultiObjFitness(values, weights=(-1., -1.))

    assert dispatcher.dispatch(objective)([_individual()]) == []
    outcome, = dispatcher.last_results
    assert not outcome.value.fitness.valid
    assert outcome.value.metadata['evaluation_failure']['code'] == 'invalid_fitness'
    summary = dispatcher.diagnostics_recorder.snapshot().summary
    assert summary.evaluations_failed == 1
    assert summary.evaluations_succeeded == 0


@pytest.mark.parametrize('values', [(float('nan'),), (float('inf'),), (float('-inf'),), (0., float('nan'))])
def test_nonfinite_existing_fitness_is_not_reused_or_implicitly_retried(values):
    individual = _individual()
    fitness = SingleObjFitness(values[0]) if len(values) == 1 else MultiObjFitness(values, weights=(-1., -1.))
    individual.set_evaluation_result(GraphEvalResult(individual.uid, fitness, individual.graph))
    dispatcher = _dispatcher()
    with pytest.raises(ValueError, match='nonfinite fitness'):
        dispatcher.dispatch(lambda graph: pytest.fail('invalid existing fitness was retried'))([individual])


def test_uid_collision_is_rejected_before_objective():
    dispatcher = _dispatcher()
    a = _individual()
    b = replace(a, graph=OptGraph([OptNode('different')]))
    dispatcher.dispatch(_size)
    with pytest.raises(ValueError, match='share an individual uid'):
        dispatcher.evaluate_population([a, b])


def test_busy_dispatcher_rejects_overlapping_batches_and_objective_replacement():
    dispatcher = _dispatcher()
    dispatcher.dispatch(_size)
    with dispatcher._owned_batch():
        with pytest.raises(RuntimeError, match='active batch'):
            dispatcher.evaluate_population([_individual()])
        with pytest.raises(RuntimeError, match='batch'):
            dispatcher.dispatch(_fail)
    assert dispatcher.dispatch(_size)([_individual()])


def test_local_dispatcher_rejects_legacy_and_implicit_remote_execution():
    with pytest.raises(ValueError, match='explicit local'):
        LocalEvaluationDispatcher(DirectAdapter(), config=EvaluationExecutionConfig())
    with pytest.raises(ValueError, match='Remote delegation'):
        _dispatcher(delegate_evaluator=SimpleNamespace(is_enabled=True))


def test_graph_preparation_cannot_silently_skip_or_extend_the_deadline():
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    clock = SimpleNamespace(timeout=timedelta(seconds=1), seconds_from_start=0.)

    def prepare(graph):
        clock.seconds_from_start = 2.
        return PreparedEvaluationGraph(graph, canonical_json({'configured': True}))

    dispatcher = _dispatcher(graph_preparation=prepare)
    assert dispatcher.dispatch(lambda graph: pytest.fail('Expired request ran'), clock)([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'EvaluationTimeout'
    assert dispatcher.last_budget.expired_before_execution
    assert dispatcher.last_budget.remaining_search_seconds == -1.
    assert dispatcher.last_budget.execution_budget_seconds is None
    dispatcher = _dispatcher(graph_preparation=lambda graph: None)
    with pytest.raises(TypeError, match='PreparedEvaluationGraph'):
        dispatcher.dispatch(_size)([_individual()])
    with pytest.raises(ValueError, match='callable'):
        _dispatcher(graph_preparation=42)


def test_preparation_consumes_explicit_batch_timeout_without_a_search_timer(monkeypatch):
    import fedot_ind.core.optimizer.evaluation.dispatcher as module
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    elapsed = [0.]
    monkeypatch.setattr(module, 'monotonic', lambda: elapsed[0])

    def prepare(graph):
        elapsed[0] = 2.
        return PreparedEvaluationGraph(graph, canonical_json({'configured': True}))

    dispatcher = LocalEvaluationDispatcher(
        DirectAdapter(), config=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL, total_timeout_seconds=1.),
        graph_preparation=prepare)
    assert dispatcher.dispatch(lambda graph: pytest.fail('Expired batch ran'))([_individual()]) == []
    result, = dispatcher.last_results
    assert result.status.value == 'timed_out'
    assert result.error.error_type == 'EvaluationTimeout'
    assert dispatcher.last_budget.preparation_seconds == 2.
    assert dispatcher.last_budget.expired_before_execution


def _fit_ridge(pipeline):
    x = np.arange(24, dtype=float).reshape(-1, 1)
    data = fedot.create_data(x, target=2 * x[:, 0] + 1, task='regression')
    pipeline.fit(data)
    prediction = pipeline.predict(data).predict
    mse = float(torch.mean((prediction.reshape(-1) - data.target.reshape(-1)) ** 2))
    assert np.isfinite(mse)
    return SingleObjFitness(mse)


def _unfit(pipeline):
    pipeline.unfit()


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
@pytest.mark.parametrize('model_name', ['ridge', 'pdl_reg'])
def test_real_native_model_fits_through_local_executor_and_is_cleaned(backend, model_name):
    before = tuple(get_registered_extensions())
    adapter = PipelineAdapter()
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    with industrial_extension_scope():
        graph = adapter.adapt(PipelineBuilder().add_node(model_name).build())
    dispatcher = LocalEvaluationDispatcher(adapter, config=EvaluationExecutionConfig(backend),
                                           graph_cleanup_fn=_unfit)
    result, = dispatcher.dispatch(_fit_ridge)([Individual(graph)])
    assert result.fitness.valid
    assert result.fitness.value < 0.01
    assert all(node.fitted_operation is None for node in adapter.restore(result.graph).nodes)
    assert tuple(get_registered_extensions()) == before
