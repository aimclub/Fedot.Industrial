import json
from shutil import copytree
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

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


def _frozen_fixture(tmp_path, *, version, sources):
    from benchmark.industrial.experiments.mutation_quality.freeze import seal, write_json
    from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
    base, bundle = tmp_path / 'base', tmp_path / 'new'
    (base / 'inputs').mkdir(parents=True)
    (base / 'inputs/example.npz').write_bytes(b'fixture bytes; verifier must not load or train')
    write_json(base / 'lock.json', {})
    seal(base)
    bundle.mkdir()
    write_json(bundle / 'environment.json', {'python': 'fixture'})
    lock = dict(schema_version=version, official_test_used=False, datasets={'example': {}}, sources=sources,
                platform_profile={'name': 'fixture'}, environment_sha256=file_digest(bundle / 'environment.json'),
                base=dict(directory=str(base), seal=file_digest(base / 'seal.json'),
                          lock_sha256=file_digest(base / 'lock.json')))
    write_json(bundle / 'lock.json', lock)
    seal(bundle)
    return base, bundle, file_digest(bundle / 'seal.json')


def test_portable_seal_checks_the_executing_copy_not_original_paths(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.cyclic_mutations import freezing, provenance
    from benchmark.industrial.experiments.mutation_quality.preflight import file_digest
    name = 'industrial/fedot_ind/example.py'
    original, active = tmp_path / 'old.py', tmp_path / 'active.py'
    original.write_text('ORIGINAL = True', encoding='utf-8')
    active.write_bytes(original.read_bytes())
    base, bundle, identity = _frozen_fixture(tmp_path, version='industrial_bd_lock@2',
                                             sources={name: file_digest(original)})
    monkeypatch.setattr(provenance, 'source_inventory', lambda: {name: active})
    monkeypatch.setattr(freezing, 'platform_profile', lambda: {'name': 'fixture'})
    checked_lock = freezing.checked_lock
    assert checked_lock(bundle, identity) == json.loads((bundle / 'lock.json').read_text())
    active.write_text('CHANGED = True', encoding='utf-8')
    with pytest.raises(ValueError, match='Active B/D source differs'):
        checked_lock(bundle, identity)
    active.write_bytes(original.read_bytes())
    moved = tmp_path / 'relocated_base'
    copytree(base, moved)
    (base / 'inputs/example.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='TRAIN input changed'):
        checked_lock(bundle, identity)
    assert checked_lock(bundle, identity, base_bundle=moved)['sources'][name] == file_digest(active)


def test_legacy_absolute_seal_can_preview_but_not_execute(tmp_path):
    from benchmark.industrial.experiments.cyclic_mutations.freezing import checked_lock
    _, bundle, identity = _frozen_fixture(tmp_path, version='industrial_bd_lock@1',
                                          sources={'old absolute path': {}})
    assert checked_lock(bundle, identity, check_current=False)['schema_version'] == 'industrial_bd_lock@1'
    with pytest.raises(ValueError, match='read-only'):
        checked_lock(bundle, identity)


def test_freeze_writes_portable_manifest_without_evaluating_models(tmp_path, monkeypatch):
    from zipfile import ZipFile
    import torch
    from benchmark.industrial.experiments.cyclic_mutations import freezing, provenance
    from benchmark.industrial.experiments.mutation_quality.freeze import seal, write_json
    from benchmark.industrial.experiments.mutation_quality.preflight import file_digest

    base, bundle = tmp_path / 'base', tmp_path / 'new'
    (base / 'inputs').mkdir(parents=True)
    tasks = {'ECG200': 'classification', 'Tecator': 'regression', 'Airline': 'ts_forecasting'}
    flags = dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic=torch.are_deterministic_algorithms_enabled())
    original = dict(git_revision_context='base-revision', execution={'torch_flags': flags}, scoring={},
                    protocol=dict(seeds=[42], budget_seconds=600, minimum_generations=15),
                    official_test_used=False, datasets={name: {} for name in tasks},
                    jobs=[dict(scenario=name, task=task) for name, task in tasks.items()])
    for name in tasks:
        (base / 'inputs' / f'{name}.npz').write_bytes(b'fixture')
    write_json(base / 'lock.json', original)
    seal(base)
    original_hash = file_digest(base / 'lock.json')
    source = tmp_path / 'source.py'
    source.write_bytes(b'VALUE = True\n')
    key = 'industrial/fedot_ind/source.py'

    def inventory():
        return {key: source}

    monkeypatch.setattr(provenance, 'source_inventory', inventory)
    monkeypatch.setattr(freezing, 'source_inventory', inventory)
    monkeypatch.setattr(freezing, 'platform_profile', lambda: {'name': 'fixture'})
    monkeypatch.setattr(freezing, 'verify_freeze', lambda *a, **k: None)
    monkeypatch.setattr(freezing, 'load_frozen_data', lambda *a: SimpleNamespace(folds=(), test=None))
    monkeypatch.setattr(freezing, 'validate_training_bounds', lambda *a: None)
    monkeypatch.setattr(freezing.subprocess, 'check_output', lambda *a, **k: 'revision')

    identity = freezing.freeze(base, file_digest(base / 'seal.json'), bundle)
    lock = freezing.checked_lock(bundle, identity)

    assert lock['schema_version'] == 'industrial_bd_lock@2' and lock['sources'] == {key: file_digest(source)}
    assert lock['environment_sha256'] == file_digest(bundle / 'environment.json')
    assert len(lock['jobs']) == 6 and lock['official_test_used'] is False
    assert file_digest(base / 'lock.json') == original_hash
    with ZipFile(bundle / 'sources.zip') as archive:
        assert archive.read(f'sources/{key}') == source.read_bytes()


def test_source_manifest_cannot_reference_paths_outside_inventory(tmp_path, monkeypatch):
    from benchmark.industrial.experiments.cyclic_mutations import provenance
    monkeypatch.setattr(provenance, 'source_inventory', lambda: {'industrial/safe.py': tmp_path / 'safe.py'})
    with pytest.raises(ValueError, match='membership'):
        provenance.verify_active_sources({'industrial/../other.py': 'trusted'})


def test_already_loaded_fedot_module_cannot_come_from_another_copy(tmp_path, monkeypatch):
    import sys
    from benchmark.industrial.experiments.cyclic_mutations.provenance import _check_loaded_modules
    monkeypatch.setitem(sys.modules, 'synthetic.wrong', SimpleNamespace(__file__=tmp_path / 'old' / 'module.py'))
    with pytest.raises(ValueError, match=r'Loaded module synthetic\.wrong'):
        _check_loaded_modules({'synthetic': tmp_path / 'active'})


def test_blocked_full_series_only_continues_verified_suffix(tmp_path, monkeypatch):
    import psutil
    from benchmark.industrial.experiments.cyclic_mutations import cleanup, execution
    from benchmark.industrial.experiments.cyclic_mutations.provenance import runtime_record
    from benchmark.industrial.experiments.evolution_parallelism import supervisor

    root = tmp_path / 'run'
    rows = [dict(scenario='Synthetic', task='regression', arm='cyclic', seed=seed, search_seconds=600,
                 minimum_generations=15, full_process_seconds=1200) for seed in (42, 123, 777)]
    lock = dict(jobs=rows, base={'directory': str(tmp_path)}, sources={}, environment_sha256='env',
                platform_profile={}, git_revision_context='fixture')
    plan = trials(lock, False)
    launched, alive = [], {(102, 20.)}
    monkeypatch.setattr(execution, 'checked_lock', lambda *a, **k: lock)
    monkeypatch.setattr(execution, 'verify_admission', lambda *a: 'gate')
    monkeypatch.setattr(execution, 'validate', lambda *a: {'status': 'succeeded', 'smoke_passed': True})
    monkeypatch.setattr(cleanup, '_worker_process_alive', alive.__contains__)

    def launch(command, **kwargs):
        index = int(command[command.index('--worker') + 1])
        launched.append(index)
        directory = root / 'trials' / plan[index].key
        directory.mkdir()
        if index == 1:
            execution.atomic_json(directory / 'worker_failure.json', dict(
                schema_version='industrial_bd_cleanup_failure@1', host=cleanup.gethostname(),
                boot_time=psutil.boot_time(), owner=[101, 10.],
                shutdown=dict(cluster_closed=True, inventory_complete=True, workers=[[102, 20.]])))
            return 2
        execution.atomic_json(directory / 'runtime.json', runtime_record(lock, 'seal'))
        return 0

    monkeypatch.setattr(supervisor, 'run_guarded', launch)
    with pytest.raises(execution.ProtocolViolation, match='remains blocked'):
        supervise(root, tmp_path, 'seal', False, gate=tmp_path)
    receipt_path = root / 'receipts' / '001.json'
    original = receipt_path.read_bytes()
    with pytest.raises(execution.ProtocolViolation, match='still alive'):
        supervise(root, tmp_path, 'seal', False, gate=tmp_path, resume_after_cleanup=1)
    assert launched == [0, 1]
    alive.clear()
    supervise(root, tmp_path, 'seal', False, gate=tmp_path, resume_after_cleanup=1)
    supervise(root, tmp_path, 'seal', False, gate=tmp_path)
    assert launched == [0, 1, 2]
    assert receipt_path.read_bytes() == original
    assert [row['exit_code'] for row in json.loads((root / 'jobs.json').read_text('utf-8'))] == [0, 2, 0]
    assert (root / 'cleanup_confirmations' / receipt_path.name).is_file()


def test_cleanup_cannot_be_confirmed_with_missing_cluster_proof(tmp_path):
    import psutil
    from benchmark.industrial.experiments.cyclic_mutations import cleanup, execution
    failure = tmp_path / 'worker_failure.json'
    execution.atomic_json(failure, dict(schema_version='industrial_bd_cleanup_failure@1', host=cleanup.gethostname(),
                                        boot_time=psutil.boot_time(), owner=[101, 10.], shutdown=None))
    with pytest.raises(execution.ProtocolViolation, match='inventory was not confirmed'):
        cleanup.confirmed_cleanup(tmp_path, tmp_path / 'receipt.json',
                                  {'exit_code': 2, 'files': {failure.name: 'hash'}})
