"""Real TensorData fold evaluation through both explicit local executors."""

from dataclasses import replace
from functools import partial
import json
import os

import numpy as np
import torch
import cv2  # noqa: F401
import pytest
import fedot
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluateWithTensorData
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _array(value):
    return value.detach().cpu().numpy()


def _fold_source(folds):
    return iter(folds)


def _metric(task, audit_path, reject_second, pipeline, reference_data, **kwargs):
    prediction = pipeline.predict(replace(reference_data, target=None, state='predict'),
                                  output_mode='labels' if task == 'classification' else 'default')
    target = _array(reference_data.target).reshape(-1)
    predicted = _array(prediction.predict).reshape(-1)
    assert target.shape == predicted.shape
    assert np.isfinite(predicted).all()
    record = {'target': target.tolist(), 'prediction': predicted.tolist(), 'pid': os.getpid()}
    with audit_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    if reject_second and len(audit_path.read_text(encoding='utf-8').splitlines()) == 2:
        raise ValueError('deliberate second fold rejection')
    if task == 'classification':
        return float(np.mean(target != predicted))
    return float(np.mean(np.abs(target - predicted)))


def _folds(task):
    result = []
    for shift in (0, 3):
        if task == 'ts_forecasting':
            from fedot.core.data.split.temporal import temporal_tensor_holdout
            from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
            values = np.sin(np.arange(64) / 5.) + np.arange(64) / 20. + shift
            data = fedot.create_data(values, target=values,
                                     task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(3)),
                                     ts_forecast_horizon=3, without_target=True)
            result.append(temporal_tensor_holdout(data))
        else:
            x = np.arange(24, dtype=float) / 8. - 1.5 + shift / 10.
            features = np.column_stack((x, np.sin(x)))
            target = (x > 0).astype(int) if task == 'classification' else 2. * x + np.cos(x) + shift
            # Both partitions contain several classes, rather than contiguous labels.
            valid = np.arange(0, 24, 4)
            train = np.setdiff1d(np.arange(24), valid)
            result.append(tuple(fedot.create_data(features[indices], target=target[indices], task=task)
                                for indices in (train, valid)))
    return tuple(result)


def _pipeline(task):
    if task == 'classification':
        return PipelineBuilder().add_node('dt', params={'max_depth': 2, 'random_state': 42}).build()
    if task == 'regression':
        return PipelineBuilder().add_node('ridge').build()
    return PipelineBuilder().add_node('hankelisation').add_node('ridge').build()


def _unfit(pipeline):
    pipeline.unfit()


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
@pytest.mark.parametrize('reject_second', [False, True])
def test_native_two_fold_evaluation_is_atomic_and_cleans_fitted_state(tmp_path, backend, task, reject_second):
    from fedot.extensions import get_registered_extensions
    before_registry = tuple(get_registered_extensions())
    folds = _folds(task)
    snapshots = [(data.features.clone(), data.target.clone()) for pair in folds for data in pair]
    audit_path = tmp_path / 'folds.jsonl'
    evaluator = PipelineObjectiveEvaluateWithTensorData(
        MetricsObjective(partial(_metric, task, audit_path, reject_second)),
        partial(_fold_source, folds), expected_folds=2, result_cache_size=0, eval_n_jobs=1)
    adapter = PipelineAdapter()
    with industrial_extension_scope():
        graph = adapter.adapt(_pipeline(task))
    individual = Individual(graph)
    recorder = EvolutionDiagnosticsRecorder()
    dispatcher = LocalEvaluationDispatcher(
        adapter, n_jobs=1, config=EvaluationExecutionConfig(backend),
        graph_cleanup_fn=_unfit, diagnostics_recorder=recorder, random_seed=42)

    evaluated = dispatcher.dispatch(evaluator.evaluate)([individual])

    records = [json.loads(line) for line in audit_path.read_text(encoding='utf-8').splitlines()]
    assert len(records) == 2
    assert all((row['pid'] == os.getpid()) == (backend is EvaluationBackend.SEQUENTIAL) for row in records)
    assert records[0]['target'] != records[1]['target']
    summary = recorder.snapshot().summary
    result, = dispatcher.last_results
    if reject_second:
        assert evaluated == []
        assert not individual.fitness.valid
        assert not result.value.fitness.valid
        failure = result.value.metadata['evaluation_failure']
        assert failure['code'] == 'objective_incomplete'
        assert failure['error_type'] == 'ValueError'
        assert 'deliberate second fold rejection' in failure['message']
        assert summary.evaluations_failed == 1
        assert summary.evaluations_succeeded == 0
    else:
        losses = []
        for record in records:
            y, prediction = np.asarray(record['target']), np.asarray(record['prediction'])
            losses.append(np.mean(y != prediction) if task == 'classification' else np.mean(np.abs(y - prediction)))
        assert evaluated == [individual]
        assert individual.fitness.value == pytest.approx(float(np.mean(losses)))
        assert summary.evaluations_succeeded == 1
        assert summary.evaluations_failed == 0
    returned = adapter.restore(result.value.graph)
    assert all(node.fitted_operation is None for node in returned.nodes)
    for data, (features, target) in zip((data for pair in folds for data in pair), snapshots):
        assert torch.equal(data.features, features)
        assert torch.equal(data.target, target)
    assert tuple(get_registered_extensions()) == before_registry
