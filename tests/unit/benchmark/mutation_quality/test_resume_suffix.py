import json

import psutil
import pytest

from benchmark.industrial.experiments.mutation_quality.execution import runner
from benchmark.industrial.experiments.mutation_quality.execution.plans import Trial


def test_suffix_preserves_frozen_indexes_and_does_not_retrain_old_jobs(tmp_path, monkeypatch):
    jobs = tuple(Trial(i, 'synthetic', 'regression', 'baseline', 42, 600, 15, 1200, False) for i in range(225))
    monkeypatch.setattr(runner, 'checked_lock', lambda *a: {'protocol': {'native_threads': 1}})
    monkeypatch.setattr(runner, 'trial_plan', lambda *a, **k: jobs)
    monkeypatch.setattr(runner, 'verify_gate', lambda *a: 'gate')
    monkeypatch.setattr(runner, 'validate_result', lambda *a: {'status': 'succeeded'})
    calls = []

    def fake_worker(command, **kwargs):
        index = int(command[command.index('--worker') + 1])
        calls.append(index)
        return 0
    monkeypatch.setattr('benchmark.industrial.experiments.evolution_parallelism.supervisor.run_guarded', fake_worker)
    rows = runner.supervise(tmp_path, 'unused', 'seal', smoke=False, gate='gate', start_index=33)
    assert calls == list(range(33, 225)) and len(rows) == 192
    assert [r['trial']['index'] for r in rows] == list(range(33, 225))
    assert len(list((tmp_path / 'receipts').glob('*.json'))) == 192
    runner.supervise(tmp_path, 'unused', 'seal', smoke=False, gate='gate', start_index=33)
    assert len(calls) == 192
    with pytest.raises(runner.ProtocolViolation, match='suffix'):
        runner.supervise(tmp_path, 'unused', 'seal', smoke=False, gate='gate', start_index=34)


def test_live_owner_cannot_be_displaced_with_recovery_flag(tmp_path):
    with runner.ownership(tmp_path):
        with pytest.raises(runner.ProtocolViolation, match='alive'):
            with runner.ownership(tmp_path, recover_stale=True):
                pytest.fail('live owner displaced')


def test_recovery_archives_stale_identity_instead_of_deleting_it(tmp_path):
    value = dict(pid=psutil.Process().pid, create_time=-1.)
    (tmp_path / 'supervisor.lock').write_text(json.dumps(value))
    with runner.ownership(tmp_path, recover_stale=True):
        archived, = tmp_path.glob('supervisor.abandoned.*.json')
        assert json.loads(archived.read_text()) == value
        assert json.loads((tmp_path / 'supervisor.lock').read_text())['create_time'] > 0


def test_controller_exception_gets_terminal_status(tmp_path):
    runner.atomic_json(tmp_path / 'status.json', {'status': 'running', 'completed': 3})
    with pytest.raises(RuntimeError):
        with runner.terminal_status(tmp_path):
            raise RuntimeError('read failed')
    row = json.loads((tmp_path / 'status.json').read_text())
    assert row['status'] == 'controller_failed' and row['completed'] == 3


@pytest.mark.parametrize('exit_code', [0x40010004, -15])
def test_external_interruption_stops_after_saved_receipt(tmp_path, monkeypatch, exit_code):
    jobs = tuple(Trial(i, 'synthetic', 'regression', 'baseline', 42, 600, 15, 1200, False) for i in range(2))
    monkeypatch.setattr(runner, 'checked_lock', lambda *a: {'protocol': {'native_threads': 1}})
    monkeypatch.setattr(runner, 'trial_plan', lambda *a, **k: jobs)
    monkeypatch.setattr(runner, 'verify_gate', lambda *a: 'gate')
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(True)
        return exit_code
    monkeypatch.setattr('benchmark.industrial.experiments.evolution_parallelism.supervisor.run_guarded', interrupted)
    with pytest.raises(runner.ProtocolViolation, match='External'):
        runner.supervise(tmp_path, 'unused', 'seal', smoke=False, gate='gate')
    assert calls == [True]
    assert (tmp_path / 'receipts/000.json').exists()
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'interrupted'
