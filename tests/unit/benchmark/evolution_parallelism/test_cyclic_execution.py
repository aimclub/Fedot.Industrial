from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import json

from benchmark.industrial.experiments.cyclic_mutations.execution import trials, supervise
from benchmark.industrial.experiments.cyclic_mutations.profile import validate_training_bounds, parameter_profile


def test_new_matrix_is_paired_and_admission_has_both_arms_per_task():
    names = dict(ECG200='classification', Tecator='regression', Airline='ts_forecasting')
    rows = [dict(scenario=n, task=task, arm=arm, seed=seed, search_seconds=600,
                 minimum_generations=15, full_process_seconds=1200)
            for n, task in names.items() for seed in (42, 123, 2026, 777, 2027) for arm in ('fixed', 'cyclic')]
    lock = dict(jobs=rows, admission_scenarios=list(names), protocol=dict(seeds=[42, 123, 2026, 777, 2027]))
    full, gate = trials(lock, False), trials(lock, True)
    assert len(full) == 30 and len(gate) == 6
    assert len({(r.task, r.arm) for r in gate}) == 6
    assert all(r.generations == 15 and r.seconds == 600 and r.smoke for r in gate)
    assert all(not r.smoke for r in full)
    lock['admission_scenarios'] = ['Tecator', 'ECG200', 'Airline']
    assert [r.scenario for r in trials(lock, True)] == ['Tecator', 'Tecator', 'ECG200', 'ECG200', 'Airline', 'Airline']
    assert trials(lock, False) == full


@pytest.mark.parametrize('code', [-15, 2, 256, 3221225477])
def test_infrastructure_and_windows_crashes_stop_full_series(code):
    from benchmark.industrial.experiments.cyclic_mutations.execution import must_stop
    assert must_stop(code, False, None)


def test_any_failed_admission_stops_and_valid_full_results_continue():
    from benchmark.industrial.experiments.cyclic_mutations.execution import must_stop
    assert must_stop(1, True, None)
    assert must_stop(0, True, {'smoke_passed': False})
    assert not must_stop(0, False, {'status': 'succeeded'})


def test_chained_execution_never_reaches_full_after_failed_admission(tmp_path):
    from benchmark.industrial.experiments.cyclic_mutations.__main__ import main
    argv = ['bd', '--run', '--admission-first', '--gate', str(tmp_path / 'gate'),
            '--bundle', str(tmp_path), '--seal', 'trusted', '--output-dir', str(tmp_path / 'full'), '--execute']
    with patch('sys.argv', argv), patch(
            'benchmark.industrial.experiments.cyclic_mutations.freezing.checked_lock', return_value={}), patch(
            'benchmark.industrial.experiments.cyclic_mutations.execution.supervise', side_effect=RuntimeError('gate failed')) as run:
        with pytest.raises(RuntimeError, match='gate failed'):
            main()
    assert run.call_count == 1 and run.call_args.args[-1] is True


def test_missing_gate_blocks_full_before_creation(tmp_path):
    with patch('benchmark.industrial.experiments.cyclic_mutations.execution.checked_lock', return_value={}):
        with pytest.raises(RuntimeError, match='requires'):
            supervise(tmp_path / 'absent', tmp_path, 'seal', False)
    assert not (tmp_path / 'absent').exists()


def test_hankel_bounds_use_actual_train_and_reject_clamping():
    task = SimpleNamespace(task_type=SimpleNamespace(name='ts_forecasting'),
                           task_params=SimpleNamespace(forecast_length=12))
    data = SimpleNamespace(refit=SimpleNamespace(task=task), folds=[(
        SimpleNamespace(task=task, features=np.arange(40).reshape(1, -1)), None)])
    with pytest.raises(ValueError, match='clamping'):
        validate_training_bounds(data, parameter_profile())
    data.folds[0][0].features = np.arange(100).reshape(1, -1)
    validate_training_bounds(data, parameter_profile())


def test_new_seal_checks_current_sources_and_archived_inputs(tmp_path):
    from benchmark.industrial.experiments.cyclic_mutations.freezing import checked_lock
    from benchmark.industrial.experiments.mutation_quality.freeze import seal, write_json
    from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
    base, bundle = tmp_path / 'base', tmp_path / 'new'
    (base / 'inputs').mkdir(parents=True)
    (base / 'inputs/example.npz').write_bytes(b'fixture bytes; verifier must not load or train')
    write_json(base / 'lock.json', {})
    seal(base)
    source = tmp_path / 'source.py'
    source.write_text('ORIGINAL = True', encoding='utf-8')
    bundle.mkdir()
    lock = dict(schema_version='industrial_bd_lock@1', official_test_used=False, datasets={'example': {}},
                sources={str(source): dict(sha256=file_digest(source))},
                base=dict(directory=str(base), seal=file_digest(base / 'seal.json'),
                          lock_sha256=file_digest(base / 'lock.json')))
    write_json(bundle / 'lock.json', lock)
    seal(bundle)
    identity = file_digest(bundle / 'seal.json')
    assert checked_lock(bundle, identity) == json.loads((bundle / 'lock.json').read_text())
    source.write_text('CHANGED = True', encoding='utf-8')
    with pytest.raises(ValueError, match='Current source'):
        checked_lock(bundle, identity)
    source.write_text('ORIGINAL = True', encoding='utf-8')
    (base / 'inputs/example.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='TRAIN input changed'):
        checked_lock(bundle, identity)
