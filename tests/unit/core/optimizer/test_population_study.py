from copy import deepcopy
import json

import numpy as np
import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.experiments.mutation_planning.population_study.protocol import (
    experiment_matrix, load_protocol, parse_protocol,
)
from benchmark.industrial.experiments.mutation_planning.population_study.report import assess_study
from benchmark.industrial.experiments.mutation_planning.population_study.runtime_contract import expected_runtime


def _records(protocol):
    from benchmark.industrial.experiments.mutation_planning.population_study.preparation import selection_summary
    from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
    configuration = {'fixture': True}
    identity = PreparedEvaluationGraph(None, canonical_json(configuration)).configuration_id
    calls = [{'candidate_id': str(i), 'evaluation_id': f'{i}:0', 'configuration': configuration,
              'configuration_id': identity, 'seed': 42, 'valid': True, 'fold_count': 1, 'fitness': score}
             for i, score in enumerate([.1, .2, .3, .05])]
    contexts = [{key: value for key, value in call.items() if key not in ('valid', 'fold_count', 'fitness')}
                for call in calls]
    entries = [{'evaluation_id': call['evaluation_id'], 'candidate_id': call['candidate_id'],
                'fitness': [call['fitness']], 'evaluation_context': context}
               for call, context in zip(calls, contexts)]
    members = [{key: call[key] for key in ('candidate_id', 'evaluation_id', 'fitness')} for call in calls[:3]]
    generations = [{'population_size': 3, 'valid_fitness_count': 3, 'new_successes': 1,
                    'members': deepcopy(members)} for _ in range(4)]
    selected = {'candidate_id': '3', 'fitness': .05, 'configuration': configuration,
                'configuration_id': identity, 'evaluation_context': contexts[3]}
    selected.update(selection_summary(selected, calls, generations, completed_entries=entries))
    return [{**row, 'status': 'succeeded', 'quality_loss': 0.1, 'baseline_loss': 0.5,
             'search_seconds': 5., 'refit_seconds': 1., 'predict_seconds': 0.01, 'peak_rss_bytes': 100000,
             'source': {'test_fixture': True}, **expected_runtime(),
             'threadpools': [{'num_threads': 1}], 'objective_calls': deepcopy(calls),
             'diagnostics': {'evaluations_succeeded': 4, 'evaluations_failed': 0},
             'completed_archive': deepcopy(entries), 'selection': deepcopy(selected),
             'selected_evaluation': deepcopy(contexts[3]), 'generations': deepcopy(generations)}
            for row in experiment_matrix(protocol)]


def test_matrix_is_complete_counterbalanced_and_only_mode_changes_within_pair():
    protocol = load_protocol()
    rows = experiment_matrix(protocol)
    assert len(rows) == 54
    assert abs(sum(row['mode'] == 'planned' for row in rows[::2]) - len(rows) / 4) <= .5
    for left, right in zip(rows[::2], rows[1::2]):
        assert {left['mode'], right['mode']} == {'planned', 'legacy'}
        assert {k: v for k, v in left.items() if k not in ('mode', 'ordinal')} == {
            k: v for k, v in right.items() if k not in ('mode', 'ordinal')}


def test_protocol_round_trip_and_caller_ownership():
    protocol = load_protocol()
    raw = protocol.to_record()
    assert parse_protocol(json.loads(json.dumps(raw))) == protocol
    raw['seeds'].append(0)
    assert len(protocol.to_record()['seeds']) == 3


@pytest.mark.parametrize('change', [
    {'seeds': [42]}, {'generations': 2}, {'population_size': False}, {'search_seconds': 0},
    {'scenarios': []}, {'extra': 1}, {'schema_version': 'other'},
])
def test_invalid_study_protocol_is_rejected(change):
    with pytest.raises(ValueError):
        parse_protocol({**load_protocol().to_record(), **change})


def test_okhs_cannot_enter_the_allowlist():
    raw = load_protocol().to_record()
    raw['operations']['ts_forecasting'].append('okhs_fdmd_forecaster')
    with pytest.raises(ValueError, match='excluded'):
        parse_protocol(raw)


@given(st.lists(st.integers(), min_size=54, max_size=54))
def test_healthy_report_is_input_order_independent(keys):
    protocol = load_protocol()
    rows = _records(protocol)
    permuted = [row for _, _, row in sorted(zip(keys, range(54), rows))]
    assert assess_study(protocol, permuted) == assess_study(protocol, rows)
    assert assess_study(protocol, rows)['status'] == 'accepted'


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failure', 'nan', 'source', 'population',
                                    'reuse', 'cost', 'unobserved_failure'])
def test_bad_or_missing_evidence_cannot_close_the_issue(damage):
    protocol = load_protocol()
    rows = _records(protocol)
    if damage == 'missing':
        rows.pop()
    elif damage == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif damage == 'failure':
        rows[0]['status'] = 'failed'
    elif damage == 'nan':
        rows[0]['quality_loss'] = float('nan')
    elif damage == 'source':
        rows[1]['source'] = {'different': True}
    elif damage == 'population':
        rows[0]['generations'][1]['population_size'] = 1
    elif damage == 'reuse':
        rows[0]['generations'][1]['new_successes'] = 0
    elif damage == 'unobserved_failure':
        rows[0]['diagnostics']['evaluations_failed'] = 1
    else:
        rows[1]['search_seconds'] = 100
    assert assess_study(protocol, rows)['status'] != 'accepted'


def test_quality_and_baseline_guards_are_not_vacuous():
    protocol = load_protocol()
    rows = _records(protocol)
    for row in rows:
        row['quality_loss'] = 0.5
    assert assess_study(protocol, rows)['status'] == 'rejected'
    rows = _records(protocol)
    for row in rows:
        if row['mode'] == 'planned':
            row['quality_loss'] = .9
    assert assess_study(protocol, rows)['status'] == 'rejected'


def test_forecast_split_never_exposes_future_values_to_training(tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study.data import load_study_data
    raw = load_protocol().to_record()
    train, test = tmp_path / 'train.csv', tmp_path / 'test.csv'
    train.write_text('id,values\nD1,' + ','.join(map(str, range(30))) + '\n', encoding='utf-8')
    test.write_text('D1,30,31,32,33\n', encoding='utf-8')
    raw.update(forecast_train=str(train), forecast_test=str(test))
    first = load_study_data(raw, {'name': 'D1', 'task': 'ts_forecasting'}, 42)
    test.write_text('D1,900,901,902,903\n', encoding='utf-8')
    second = load_study_data(raw, {'name': 'D1', 'task': 'ts_forecasting'}, 42)
    for field in ('train', 'validation', 'refit'):
        np.testing.assert_equal(getattr(first, field).features.numpy(), getattr(second, field).features.numpy())
        np.testing.assert_equal(getattr(first, field).target.numpy(), getattr(second, field).target.numpy())
    assert first.scale == second.scale
    assert first.source['split'] == {'train_end': 26, 'validation_end': 30, 'test_end': 34}
    np.testing.assert_equal(first.test.target.numpy().reshape(-1), [30, 31, 32, 33])


def test_cpu_admission_rejects_visible_cuda_before_data_or_training(monkeypatch, tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study import worker
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(worker, 'load_study_data', lambda *args: pytest.fail('Data was loaded before admission'))
    with pytest.raises(RuntimeError, match='admission'):
        worker.run_trial(load_protocol().to_record(), experiment_matrix(load_protocol())[0], tmp_path)


def test_worker_environment_hides_cuda_with_explicit_windows_safe_value(monkeypatch, tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    seen = {}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('PYTORCH_NVML_BASED_CUDA_CHECK', '0')

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            seen.update(kwargs['env'])
            seen['command'] = command

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    assert runner.execute_worker(tmp_path / 'job.json', 1) == 0
    assert seen['CUDA_VISIBLE_DEVICES'] == '-1'
    assert seen['PYTORCH_NVML_BASED_CUDA_CHECK'] == '1'
    assert seen['command'][1:5] == ['-u', '-X', 'faulthandler', '-m']
    assert all(seen[key] == '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    assert runner.os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert runner.os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] == '0'


@pytest.mark.parametrize('message', [
    'Windows fatal exception: access violation', 'Fatal Python error: Aborted',
])
def test_native_fault_in_worker_log_is_not_clean_evidence(tmp_path, message):
    from benchmark.industrial.experiments.mutation_planning.population_study.runner import native_crash_message
    log = tmp_path / 'run.log'
    log.write_text('search-finished\n' + message + '\nThread ...\n', encoding='utf-8')
    assert native_crash_message(log) == message


def test_ordinary_worker_log_does_not_imply_a_native_fault(tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study.runner import native_crash_message
    log = tmp_path / 'run.log'
    log.write_text('search-finished\nNo Fatal Python error: recorded\n', encoding='utf-8')
    assert native_crash_message(log) is None


@pytest.mark.parametrize('interrupted', [False, True])
def test_coordinator_preserves_but_rejects_published_result_after_native_fault(monkeypatch, tmp_path, interrupted):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    protocol = load_protocol()
    job = experiment_matrix(protocol)[0]
    key = f"{job['scenario']['name']}:{job['seed']}"
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {})
    monkeypatch.setattr(runner, 'freeze_run', lambda *args: {
        'matrix': [job], 'protocol': protocol.to_record(), 'implementation_sha256': {},
        'data_sources': {key: {}},
    })

    def native_fault(job_file, timeout):
        runner.write_json(job_file.with_name('result.json'), {**job, 'status': 'succeeded', 'source': {}})
        job_file.with_name('run.log').write_text('Windows fatal exception: access violation\n', encoding='utf-8')
        return 0

    monkeypatch.setattr(runner, 'execute_worker', native_fault)
    if interrupted:
        write_json = runner.write_json

        def interrupt_failure_publication(path, value):
            if path.name == 'coordinator_failure.json':
                raise KeyboardInterrupt
            write_json(path, value)

        monkeypatch.setattr(runner, 'write_json', interrupt_failure_publication)
        with pytest.raises(KeyboardInterrupt):
            runner.run_study(tmp_path, worktree=tmp_path, execute=True)
        monkeypatch.setattr(runner, 'write_json', write_json)
        monkeypatch.setattr(runner, 'execute_worker', lambda *args: pytest.fail('unexpected retry'))
    report = runner.run_study(tmp_path, worktree=tmp_path, execute=True)
    assert report['status'] == 'incomplete'
    record, = json.loads((tmp_path / 'records.json').read_text(encoding='utf-8'))
    assert record['status'] == 'failed'
    assert 'native fault' in record['error_message']
    directory, = tmp_path.glob('run_*')
    assert json.loads((directory / 'process.json').read_text(encoding='utf-8'))['exit_code'] == 0
    assert json.loads((directory / 'result.json').read_text(encoding='utf-8'))['status'] == 'succeeded'
    # Resume consumes the explicit failure; it must not silently retry or accept the raw result.
    monkeypatch.setattr(runner, 'execute_worker', lambda *args: pytest.fail('unexpected retry'))
    assert runner.run_study(tmp_path, worktree=tmp_path, execute=True) == report


def test_source_snapshot_includes_packaged_quantile_batch_limits():
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    from benchmark.industrial.experiments.mutation_planning.population_study.data import file_hash
    from pathlib import Path
    name = 'fedot_ind/core/operation/transformation/torch_backend/statistical/batch_defaults.json'
    root = Path(runner.__file__).resolve().parents[5]
    assert runner._source_snapshot()[name] == file_hash(root / name)


@pytest.mark.parametrize('damage', ['missing', 'determinism', 'matmul', 'build', 'bool_as_int'])
def test_frozen_torch_policy_is_part_of_acceptance(damage):
    protocol = load_protocol()
    rows = _records(protocol)
    if damage == 'missing':
        del rows[0]['torch_flags']
    elif damage == 'determinism':
        rows[0]['torch_flags']['deterministic_algorithms'] = True
    elif damage == 'matmul':
        rows[0]['torch_flags']['float32_matmul_precision'] = 'high'
    elif damage == 'build':
        rows[0]['torch_flags']['torch_cuda_build'] = '12.4'
    else:
        rows[0]['torch_flags']['cudnn_benchmark'] = 0
    report = assess_study(protocol, rows)
    assert report['status'] == 'rejected'
    assert not report['pairs'][0]['checks']['legacy_frozen_torch']


def test_cuda_build_is_rejected_without_initializing_cuda(monkeypatch):
    from types import SimpleNamespace
    from benchmark.industrial.experiments.mutation_planning.population_study.runtime_contract import require_runtime
    torch = SimpleNamespace(version=SimpleNamespace(cuda='12.4'),
                            cuda=SimpleNamespace(is_available=lambda: pytest.fail('CUDA driver accessed')))
    with pytest.raises(RuntimeError, match='CPU-only'):
        require_runtime(torch)


def test_code_is_locked_before_any_dataset_read_and_drift_prevents_resume(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    import fedot
    import fedot_ind.integration.fedot.compat as compat
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    monkeypatch.setattr(compat, 'audit_compatibility_worktree', lambda path: {
        'worktree': str(Path(fedot.__file__).resolve().parents[1])})
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {'code.py': 'before'})
    reads = []

    def load(config, scenario, seed):
        lock = json.loads((tmp_path / 'code_lock.json').read_text())
        assert lock['implementation_sha256'] == {'code.py': 'before'}
        assert lock['runtime'] == expected_runtime()
        reads.append((scenario['name'], seed))
        return SimpleNamespace(source={'fixture': True})

    monkeypatch.setattr(runner, 'load_study_data', load)
    manifest = runner.freeze_run(load_protocol(), tmp_path, tmp_path)
    assert len(reads) == 27
    assert runner.file_hash(tmp_path / 'code_lock.json') == manifest['code_lock_sha256']
    assert runner.freeze_run(load_protocol(), tmp_path, tmp_path) == manifest
    reads.clear()
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {'code.py': 'after'})
    with pytest.raises(ValueError, match='Frozen code/configuration/runtime changed'):
        runner.freeze_run(load_protocol(), tmp_path, tmp_path)
    assert not reads


@pytest.mark.parametrize('last_check', [False, True])
def test_drift_after_worker_or_after_last_record_cannot_pass(monkeypatch, tmp_path, last_check):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    protocol = load_protocol()
    row = _records(protocol)[0]
    job = experiment_matrix(protocol)[0]
    source = row['source']
    calls = []

    def snapshot():
        calls.append(None)
        return {} if len(calls) < (4 if last_check else 2) else {'changed.py': 'new'}

    monkeypatch.setattr(runner, '_source_snapshot', snapshot)
    monkeypatch.setattr(runner, 'freeze_run', lambda *args: {
        'matrix': [job], 'protocol': protocol.to_record(), 'implementation_sha256': {},
        'data_sources': {f"{job['scenario']['name']}:{job['seed']}": source},
    })

    def evaluate(job_file, timeout):
        runner.write_json(job_file.with_name('result.json'), row)
        job_file.with_name('run.log').write_text('clean exit\n')
        return 0

    monkeypatch.setattr(runner, 'execute_worker', evaluate)
    with pytest.raises(ValueError, match='implementation changed'):
        runner.run_study(tmp_path, worktree=tmp_path, execute=True)
    if last_check:
        assert json.loads((tmp_path / 'report.json').read_text())['status'] != 'accepted'


@pytest.mark.parametrize('damage', ['content', 'missing'])
def test_final_integrity_failure_cannot_leave_an_accepted_report(monkeypatch, tmp_path, damage):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    protocol = load_protocol()
    rows = _records(protocol)
    jobs = experiment_matrix(protocol)
    runner.write_json(tmp_path / 'code_lock.json', {'frozen': True})
    lock_hash = runner.file_hash(tmp_path / 'code_lock.json')
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {})
    monkeypatch.setattr(runner, 'freeze_run', lambda *args: {
        'matrix': jobs, 'protocol': protocol.to_record(), 'implementation_sha256': {},
        'data_sources': {f"{row['scenario']['name']}:{row['seed']}": row['source'] for row in rows},
        'code_lock_sha256': lock_hash,
    })

    def evaluate(job_file, timeout):
        job = json.loads(job_file.read_text())['job']
        runner.write_json(job_file.with_name('result.json'), rows[job['ordinal']])
        job_file.with_name('run.log').write_text('clean exit\n')
        return 0

    write_json = runner.write_json

    def corrupt_after_positive_report(path, value):
        write_json(path, value)
        if path.name == 'report.json' and value['status'] == 'accepted':
            if damage == 'content':
                write_json(tmp_path / 'code_lock.json', {'frozen': False})
            else:
                (tmp_path / 'code_lock.json').unlink()

    monkeypatch.setattr(runner, 'write_json', corrupt_after_positive_report)
    monkeypatch.setattr(runner, 'execute_worker', evaluate)
    with pytest.raises((ValueError, OSError)):
        runner.run_study(tmp_path, worktree=tmp_path, execute=True)
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['status'] == 'incomplete' and report['observed_runs'] == 54
    assert any('code_lock' in problem or 'code lock changed' in problem for problem in report['problems'])


def test_source_snapshot_covers_the_executed_integration_layer():
    from benchmark.industrial.experiments.mutation_planning.population_study.runner import _source_snapshot
    snapshot = _source_snapshot()
    assert 'fedot_ind/integration/fedot/evaluation.py' in snapshot
    assert 'fedot_ind/integration/fedot/extensions/factories.py' in snapshot
    assert 'benchmark/industrial/experiments/mutation_planning/population_study/runtime_defaults.json' in snapshot
    assert 'fedot_ind/core/repository/constanst_repository.py' in snapshot
    assert 'fedot_ind/core/repository/IndustrialDispatcher.py' in snapshot


def test_external_fedot_drift_is_not_hidden_by_matching_compatibility_patches(monkeypatch, tmp_path):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    root = tmp_path / 'fedot'
    root.mkdir()
    source = root / 'regular_unpatched.py'
    source.write_text('VALUE = 1\n')
    snapshot = runner._fedot_source_snapshot(tmp_path)
    assert set(snapshot) == {'fedot/regular_unpatched.py'}
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {})
    manifest = {'implementation_sha256': {}, 'fedot_sources_sha256': snapshot,
                'fedot_compatibility': {'worktree': str(tmp_path)}}
    runner._verify_frozen_execution(manifest, tmp_path)
    source.write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='FEDOT implementation changed'):
        runner._verify_frozen_execution(manifest, tmp_path)


@pytest.mark.parametrize('damage', ['missing', 'omitted_best', 'count', 'selected', 'final_as_generation',
                                    'context_seed', 'context_configuration'])
def test_full_acceptance_requires_complete_archive_and_real_generations(damage):
    protocol = load_protocol()
    rows = _records(protocol)
    row = rows[0]
    if damage == 'missing':
        del row['completed_archive']
    elif damage == 'omitted_best':
        from benchmark.industrial.experiments.mutation_planning.population_study.preparation import selection_summary
        row['completed_archive'].pop()
        call = row['objective_calls'][0]
        context = row['completed_archive'][0]['evaluation_context']
        row['selected_evaluation'] = deepcopy(context)
        row['selection'] = {'candidate_id': call['candidate_id'], 'fitness': call['fitness'],
                            'configuration': call['configuration'], 'configuration_id': call['configuration_id'],
                            'evaluation_context': context}
        row['selection'].update(selection_summary(row['selection'], row['objective_calls'], row['generations'],
                                                  completed_entries=row['completed_archive']))
    elif damage == 'count':
        row['completed_archive'].pop(0)
    elif damage == 'selected':
        row['selection']['fitness'] = .001
    elif damage == 'context_seed':
        row['selection']['evaluation_context']['seed'] = 123
        row['selected_evaluation']['seed'] = 123
    elif damage == 'context_configuration':
        from fedot_ind.core.optimizer.evaluation.preparation import PreparedEvaluationGraph, canonical_json
        configuration = {'fixture': 'changed'}
        identity = PreparedEvaluationGraph(None, canonical_json(configuration)).configuration_id
        row['selection'].update(configuration=configuration, configuration_id=identity)
        for context in (row['selection']['evaluation_context'], row['selected_evaluation']):
            context.update(configuration=configuration, configuration_id=identity)
    else:
        row['generations'][3]['is_final'] = True
    report = assess_study(protocol, rows)
    assert report['status'] == 'rejected'
    check = 'legacy_full_populations' if damage == 'final_as_generation' else 'legacy_completed_archive'
    assert not report['pairs'][0]['checks'][check]


@pytest.mark.parametrize('marker', ['attempt.json', 'process.json', 'run.log'])
def test_interrupted_attempt_is_preserved_without_automatic_retry(monkeypatch, tmp_path, marker):
    from benchmark.industrial.experiments.mutation_planning.population_study import runner
    protocol = load_protocol()
    job = experiment_matrix(protocol)[0]
    directory = tmp_path / f"run_{job['ordinal']:03}_{job['scenario']['name']}_{job['seed']}_{job['mode']}"
    directory.mkdir()
    (directory / marker).write_text('{"exit_code": 1}\n', encoding='utf-8')
    monkeypatch.setattr(runner, '_source_snapshot', lambda: {})
    monkeypatch.setattr(runner, 'freeze_run', lambda *args: {
        'matrix': [job], 'protocol': protocol.to_record(), 'implementation_sha256': {}, 'data_sources': {},
    })
    monkeypatch.setattr(runner, 'execute_worker', lambda *args: pytest.fail('A prior attempt was retried'))
    report = runner.run_study(tmp_path, worktree=tmp_path, execute=True)
    assert report['status'] == 'incomplete'
    failure = json.loads((directory / 'coordinator_failure.json').read_text())
    assert failure['error_type'] == 'InterruptedAttempt'
    assert (directory / marker).read_text() == '{"exit_code": 1}\n'
    assert runner.run_study(tmp_path, worktree=tmp_path, execute=True) == report
