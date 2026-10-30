from copy import deepcopy
from dataclasses import replace, asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_quality.execution.plans import trial_plan, Trial
from benchmark.industrial.experiments.mutation_quality.execution.runner import (
    ProtocolViolation, atomic_json, receipt, verify_receipt, verify_gate, supervise, ownership,
)


def test_full_and_smoke_plans_do_not_change_frozen_jobs():
    from benchmark.industrial.experiments.mutation_quality.config import load_protocol
    protocol = load_protocol()
    tasks = {row.name: row.task for row in protocol.scenarios}
    lock = dict(protocol=asdict(protocol), datasets={row.name: {} for row in protocol.scenarios},
                jobs=[dict(scenario=name, task=tasks[name], arm=arm, seed=seed, search_seconds=600,
                           minimum_generations=15, full_process_seconds=1200) for name, arm, seed in protocol.jobs()])
    original = deepcopy(lock)
    full, smoke = trial_plan(lock, smoke=False), trial_plan(lock, smoke=True)
    assert lock == original
    assert len(full) == 225 and len(smoke) == 15
    assert {row.arm for row in smoke} == {'baseline', 'expanded_fixed', 'expanded_staged'}
    assert {row.scenario for row in smoke} == set(tasks)
    assert all(row.seconds == 600 and row.generations == 15 and not row.smoke for row in full)
    assert all(row.seconds == 600 and row.generations == 1 and row.smoke for row in smoke)


def container_metadata():
    from benchmark.industrial.experiments.mutation_quality.freeze import array_record
    arrays = {}
    roles = [f'fold{i}_{role}' for i in range(3) for role in ('train', 'validation')] + ['refit', 'control']
    metadata = dict(containers={}, arrays={}, source=dict(dataset='synthetic', official_test_used=False))
    for role in roles:
        metadata['containers'][role] = dict(state='StateEnum.FIT', data_type='DataTypesEnum.multi_ts',
                                            task='TaskTypesEnum.regression', horizon=None)
        for key, array in dict(features=np.arange(12, dtype=np.float32).reshape(4, 1, 3),
                               target=np.arange(4), idx=np.arange(4)).items():
            name = f'{role}_{key}'
            arrays[name] = array
            metadata['arrays'][name] = array_record(array)[1]
    return arrays, metadata


def test_frozen_loader_does_not_read_outer_arrays_or_raw_sources(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.mutation_quality.execution.frozen_data import load_frozen_data
    arrays, metadata = container_metadata()
    (tmp_path / 'inputs').mkdir()
    np.savez(tmp_path / 'inputs/synthetic.npz', **arrays)
    original = np.lib.npyio.NpzFile.__getitem__
    observed = []

    def tracked(self, key):
        assert not key.startswith('control_')
        observed.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', tracked)
    lock = dict(datasets={'synthetic': metadata}, protocol=dict(inner_folds=3))
    data = load_frozen_data(tmp_path, lock, 'synthetic')
    assert data.test is None and len(data.folds) == 3 and len(observed) == 21
    np.testing.assert_array_equal(data.folds[0][0].features.numpy(), arrays['fold0_train_features'])


def test_control_is_loaded_only_in_final_full_evaluation(monkeypatch):
    from benchmark.industrial.experiments.mutation_quality.execution.study import FrozenStudy
    from benchmark.industrial.experiments.evolution_parallelism.data import SearchData
    from benchmark.industrial.experiments.mutation_diversity.study import MutationStudy
    from benchmark.industrial.experiments.staged_mutations.training import StagedTrainingStudy
    data = SearchData((), None, None, {})
    events = []
    monkeypatch.setattr('benchmark.industrial.experiments.mutation_quality.execution.study.load_control',
                        lambda *args: events.append('load_control') or 'outer')
    monkeypatch.setattr(MutationStudy, 'evaluate_selected',
                        lambda self, winner, actual, root: events.append(actual.test))
    monkeypatch.setattr(StagedTrainingStudy, 'evaluate_selected',
                        lambda self, winner, actual, root: events.append('inner_only'))
    study = FrozenStudy('baseline', 'regression', pilot=True, bundle='unused', lock={},
                        trial=SimpleNamespace(scenario='synthetic'))
    study.evaluate_selected('selected', data, Path('.'))
    assert events == ['inner_only'] and data.test is None
    replace(study, pilot=False).evaluate_selected('selected', data, Path('.'))
    assert events == ['inner_only', 'load_control', 'outer'] and data.test is None


def test_receipts_detect_tampering_and_never_retrain_completed_trials(tmp_path):
    trial = Trial(0, 'synthetic', 'regression', 'baseline', 42, 600, 15, 1200, False)
    (tmp_path / 'result').write_text('done', encoding='utf-8')
    value = receipt(tmp_path, trial, 0, 'incomplete', {'generations': 3})
    verify_receipt(tmp_path, value)
    assert value['status'] == 'incomplete'
    (tmp_path / 'result').write_text('changed', encoding='utf-8')
    with pytest.raises(ProtocolViolation, match='artifacts changed'):
        verify_receipt(tmp_path, value)


def test_atomic_status_and_single_supervisor_ownership(tmp_path):
    path = tmp_path / 'status.json'
    atomic_json(path, {'state': 'first'})
    atomic_json(path, {'state': 'second'})
    assert json.loads(path.read_text('utf-8')) == {'state': 'second'}
    with ownership(tmp_path):
        with pytest.raises(FileExistsError):
            with ownership(tmp_path):
                pytest.fail('Second supervisor acquired ownership')
    assert not (tmp_path / 'supervisor.lock').exists()


def test_gate_rejects_incomplete_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'benchmark.industrial.experiments.mutation_quality.execution.runner.check_binding', lambda *a: {})
    (tmp_path / 'jobs.json').write_text('[]', encoding='utf-8')
    lock = dict(datasets={'synthetic': {}}, protocol={'arms': ['baseline'], 'seeds': [42]}, jobs=[
        dict(scenario='synthetic', task='regression', arm='baseline', seed=42,
             search_seconds=600, minimum_generations=15, full_process_seconds=1200)])
    with pytest.raises(ProtocolViolation, match='incomplete'):
        verify_gate(tmp_path, 'unused', 'unused', lock)


def test_full_supervision_requires_gate_before_training(tmp_path):
    with pytest.raises(ProtocolViolation, match='smoke gate'):
        supervise(tmp_path, 'unused', 'unused', smoke=False)
    assert list(tmp_path.iterdir()) == []


def test_supervisor_resume_does_not_repeat_completed_job(tmp_path, monkeypatch):
    import benchmark.industrial.experiments.mutation_quality.execution.runner as runner
    trial = Trial(0, 'synthetic', 'regression', 'baseline', 42, 150, 1, 420, True)
    lock = dict(protocol={'native_threads': 1})
    monkeypatch.setattr(runner, 'checked_lock', lambda *args: lock)
    monkeypatch.setattr(runner, 'trial_plan', lambda *args, **kwargs: (trial,))
    monkeypatch.setattr(runner, 'validate_result', lambda *args: dict(status='succeeded', smoke_passed=True))
    calls = []

    def fake_worker(*args, **kwargs):
        directory = tmp_path / 'trials' / trial.key
        directory.mkdir(parents=True)
        (directory / 'completed').write_text('synthetic test, not training', encoding='utf-8')
        calls.append(trial.key)
        return 0

    monkeypatch.setattr('benchmark.industrial.experiments.evolution_parallelism.supervisor.run_guarded', fake_worker)
    first = runner.supervise(tmp_path, 'unused', 'unused', smoke=True)
    second = runner.supervise(tmp_path, 'unused', 'unused', smoke=True)
    assert first == second and calls == [trial.key]


def test_environment_check_matches_post_import_freeze_and_still_rejects_changes():
    from importlib import import_module
    import_module('fedot')
    from importlib.metadata import distributions
    import sys
    from benchmark.industrial.experiments.mutation_quality.execution.runner import check_environment
    expected = dict(packages={dist.metadata['Name']: dist.version for dist in distributions()}, python=sys.version)
    check_environment(expected)
    expected['packages']['torch'] = 'not-the-frozen-version'
    with pytest.raises(ProtocolViolation, match='Dependency environment'):
        check_environment(expected)


@pytest.mark.parametrize('task', ['classification', 'regression', 'ts_forecasting'])
def test_real_final_refit_and_control_prediction_on_synthetic_data(tmp_path, task):
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    import torch
    from benchmark.industrial.experiments.evolution_parallelism.data import SearchData
    from benchmark.industrial.experiments.mutation_quality.freeze import save_data
    from benchmark.industrial.experiments.mutation_quality.execution.study import FrozenStudy
    from benchmark.industrial.experiments.staged_mutations.training import load_training_protocol
    from fedot_ind.integration.fedot.extensions import industrial_extension_scope
    settings = load_training_protocol()
    if task == 'ts_forecasting':
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        values = np.arange(132, dtype=float) + np.sin(np.arange(132) / 3)
        complete = create_data(values, target=values.copy(), task=task, ts_forecast_horizon=12, without_target=True)
        train, control = temporal_tensor_holdout(complete)
    else:
        x = np.arange(320, dtype=np.float32).reshape(20, 1, 16) / 100
        y = np.arange(20) % 2 if task == 'classification' else np.arange(20, dtype=float)

        def container(left, right):
            return TensorData(features=torch.as_tensor(x[left:right]), target=torch.as_tensor(y[left:right]),
                              idx=np.arange(right - left), task=Task(TaskTypesEnum[task]), data_type=DataTypesEnum.multi_ts)
        train, control = container(0, 16), container(16, 20)
    (tmp_path / 'inputs').mkdir()
    data = SearchData(((train, control),) * 3, train, control, {'official_test_used': False})
    metadata = save_data(tmp_path / 'inputs/synthetic.npz', data)
    data.test = None
    study = FrozenStudy('baseline', task, pilot=False, bundle=str(tmp_path), lock={'datasets': {'synthetic': metadata}},
                        trial=SimpleNamespace(scenario='synthetic'))
    with industrial_extension_scope():
        builder = PipelineBuilder()
        for operation in settings['tasks'][task]['initial_chains'][0]:
            builder.add_node(operation, params=settings['operation_parameters'].get(operation))
        pipeline = builder.build()
        final, loss = study.evaluate_selected(pipeline, data, tmp_path)
    assert np.isfinite(loss) and final.test_loss is None
    assert data.test is None and (tmp_path / 'outer_train_predictions.csv').is_file()
