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
from fedot_ind.core.optimizer.evaluation.contracts import ExecutionStatus
from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


def _array(value):
    return value.detach().cpu().numpy()


def _fold_source(folds):
    return iter(folds)


class _InputAuditedEvaluator(PipelineObjectiveEvaluateWithTensorData):
    def __init__(self, *args, input_audit_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_audit_path = input_audit_path

    def evaluate(self, graph):
        # Snapshot after deserialization, where fit/predict can mutate the actual inputs.
        data = tuple(data for pair in self.tensor_data for data in pair)
        snapshots = [(item.features.clone(), item.target.clone()) for item in data]
        try:
            return super().evaluate(graph)
        finally:
            for item, (features, target) in zip(data, snapshots):
                assert torch.equal(item.features, features)
                assert torch.equal(item.target, target)
            self.input_audit_path.write_text(
                json.dumps({'pid': os.getpid(), 'partitions_checked': len(data)}), encoding='utf-8')


def _metric(task, audit_path, reject_second, fitted_operations, pipeline, reference_data, **kwargs):
    operations = tuple(node.fitted_operation for node in pipeline.nodes)
    assert all(operation is not None for operation in operations)
    assert all(operation is not previous for previous_fold in fitted_operations
               for previous in previous_fold for operation in operations)
    # Keep strong references so Python cannot recycle object identities between folds.
    fitted_operations.append(operations)
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


def _mutating_metric(data, tensor_name, audit_path, pipeline, reference_data, **kwargs):
    assert pipeline.is_fitted
    tensor = getattr(data, tensor_name)
    before = tensor.clone()
    tensor.add_(1)
    audit_path.write_text(json.dumps({'pid': os.getpid(), 'mutated': not torch.equal(tensor, before)}),
                          encoding='utf-8')
    return 0.


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
            target = ((x > 0).astype(int) ^ int(shift > 0) if task == 'classification'
                      else 2. * x + np.cos(x) + shift)
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


def _fresh_fold_reference(task):
    references = []
    # Recreate data and pipelines independently of the evaluator and its metric callback.
    with industrial_extension_scope():
        for train, test in _folds(task):
            pipeline = _pipeline(task)
            try:
                pipeline.fit(train, n_jobs=1)
                output = pipeline.predict(replace(test, target=None, state='predict'),
                                          output_mode='labels' if task == 'classification' else 'default')
                target = _array(test.target).reshape(-1).copy()
                prediction = _array(output.predict).reshape(-1).copy()
                loss = (np.mean(target != prediction) if task == 'classification'
                        else np.mean(np.abs(target - prediction)))
                references.append((target, prediction, float(loss)))
            finally:
                pipeline.unfit()
    return references


@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
def test_native_fold_reference_detects_reusing_first_fold_model(task):
    references = _fresh_fold_reference(task)
    (train, _), (_, test) = _folds(task)
    with industrial_extension_scope():
        pipeline = _pipeline(task)
        try:
            pipeline.fit(train, n_jobs=1)
            stale = pipeline.predict(replace(test, target=None, state='predict'),
                                     output_mode='labels' if task == 'classification' else 'default')
            assert not np.allclose(_array(stale.predict).reshape(-1), references[1][1],
                                   rtol=1e-6, atol=1e-7)
        finally:
            pipeline.unfit()


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS])
@pytest.mark.parametrize('partition', [0, 1], ids=['train', 'validation'])
@pytest.mark.parametrize('tensor_name', ['features', 'target'])
def test_native_fold_input_guard_rejects_worker_local_mutation(tmp_path, backend, partition, tensor_name):
    folds = _folds('regression')[:1]
    mutation_audit_path = tmp_path / 'mutation.json'
    input_audit_path = tmp_path / 'inputs.json'
    evaluator = _InputAuditedEvaluator(
        MetricsObjective(partial(_mutating_metric, folds[0][partition], tensor_name, mutation_audit_path)),
        partial(_fold_source, folds), expected_folds=1, result_cache_size=0, eval_n_jobs=1,
        input_audit_path=input_audit_path)
    adapter = PipelineAdapter()
    with industrial_extension_scope():
        individual = Individual(adapter.adapt(_pipeline('regression')))
    recorder = EvolutionDiagnosticsRecorder()
    dispatcher = LocalEvaluationDispatcher(
        adapter, n_jobs=1, config=EvaluationExecutionConfig(backend),
        graph_cleanup_fn=_unfit, diagnostics_recorder=recorder, random_seed=42)

    assert dispatcher.dispatch(evaluator.evaluate)([individual]) == []

    mutation = json.loads(mutation_audit_path.read_text(encoding='utf-8'))
    assert mutation['mutated'] is True
    assert (mutation['pid'] == os.getpid()) == (backend is EvaluationBackend.SEQUENTIAL)
    result, = dispatcher.last_results
    assert result.status is ExecutionStatus.FAILED
    assert result.error.error_type == 'AssertionError'
    assert not individual.fitness.valid
    assert not input_audit_path.exists()
    summary = recorder.snapshot().summary
    assert summary.evaluations_failed == 1
    assert summary.evaluations_succeeded == 0


@pytest.mark.parametrize('backend', [EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS, EvaluationBackend.DASK])
@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
@pytest.mark.parametrize('reject_second', [False, True])
def test_native_two_fold_evaluation_is_atomic_and_cleans_fitted_state(tmp_path, backend, task, reject_second):
    from fedot.extensions import get_registered_extensions
    before_registry = tuple(get_registered_extensions())
    references = _fresh_fold_reference(task)
    folds = _folds(task)
    snapshots = [(data.features.clone(), data.target.clone()) for pair in folds for data in pair]
    audit_path = tmp_path / 'folds.jsonl'
    input_audit_path = tmp_path / 'inputs.json'
    evaluator = _InputAuditedEvaluator(
        MetricsObjective(partial(_metric, task, audit_path, reject_second, [])),
        partial(_fold_source, folds), expected_folds=2, result_cache_size=0, eval_n_jobs=1,
        input_audit_path=input_audit_path)
    adapter = PipelineAdapter()
    with industrial_extension_scope():
        graph = adapter.adapt(_pipeline(task))
    individual = Individual(graph)
    recorder = EvolutionDiagnosticsRecorder()
    dispatcher_type = LocalEvaluationDispatcher
    if backend is EvaluationBackend.DASK:
        from fedot_ind.core.optimizer.evaluation.dask_dispatcher import DaskEvaluationDispatcher
        dispatcher_type = DaskEvaluationDispatcher
    dispatcher = dispatcher_type(
        adapter, n_jobs=1, config=EvaluationExecutionConfig(backend),
        graph_cleanup_fn=_unfit, diagnostics_recorder=recorder, random_seed=42)

    evaluated = dispatcher.dispatch(evaluator.evaluate)([individual])

    records = [json.loads(line) for line in audit_path.read_text(encoding='utf-8').splitlines()]
    assert len(records) == 2
    assert all((row['pid'] == os.getpid()) == (backend is EvaluationBackend.SEQUENTIAL) for row in records)
    input_audit = json.loads(input_audit_path.read_text(encoding='utf-8'))
    assert input_audit == {'pid': records[0]['pid'], 'partitions_checked': 4}
    assert {record['pid'] for record in records} == {input_audit['pid']}
    for record, (target, prediction, _) in zip(records, references):
        np.testing.assert_array_equal(record['target'], target)
        if task == 'classification':
            np.testing.assert_array_equal(record['prediction'], prediction)
        else:
            np.testing.assert_allclose(record['prediction'], prediction, rtol=1e-6, atol=1e-7)
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
        assert evaluated == [individual]
        assert individual.fitness.value == pytest.approx(float(np.mean([loss for _, _, loss in references])))
        assert summary.evaluations_succeeded == 1
        assert summary.evaluations_failed == 0
    returned = adapter.restore(result.value.graph)
    assert all(node.fitted_operation is None for node in returned.nodes)
    for data, (features, target) in zip((data for pair in folds for data in pair), snapshots):
        assert torch.equal(data.features, features)
        assert torch.equal(data.target, target)
    assert tuple(get_registered_extensions()) == before_registry
