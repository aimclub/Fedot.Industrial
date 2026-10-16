from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
import cv2  # noqa: F401
import pytest
import fedot
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.extensions import get_registered_extensions
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.fitness import SingleObjFitness, null_fitness
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual

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
    dispatcher.dispatch(_size)
    dispatcher.cancel()
    assert dispatcher.evaluate_population([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'Cancelled'


def test_invalid_fitness_is_not_in_successful_population():
    dispatcher = _dispatcher()
    assert dispatcher.dispatch(lambda graph: null_fitness())([_individual()]) == []
    summary = dispatcher.diagnostics_recorder.snapshot().summary
    assert summary.evaluations_failed == 1
    assert summary.evaluations_succeeded == 0


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
