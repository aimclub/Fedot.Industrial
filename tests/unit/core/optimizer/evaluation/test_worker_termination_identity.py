from threading import Barrier
from types import SimpleNamespace

import pytest

from fedot_ind.core.optimizer.evaluation import dask_executor as module
from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig


def session_with_reply(monkeypatch, reply, alive):
    session = module.DaskEvaluationSession(1, DaskExecutionConfig())
    session.worker_processes = {'old': (123, 10.)}
    session._verified.add('old')
    session.client = SimpleNamespace(cancel=lambda *a, **k: None, restart_workers=lambda *a, **k: {'old': reply})
    monkeypatch.setattr(module, '_worker_process_alive', lambda identity: alive)
    return session


def test_replacement_timeout_is_not_confused_with_old_process_still_running(monkeypatch):
    session = session_with_reply(monkeypatch, 'timed out', False)
    session.interrupt_worker('old', object())
    assert 'old' not in session._verified


def test_ready_acknowledgement_does_not_hide_still_alive_old_fit(monkeypatch):
    session = session_with_reply(monkeypatch, 'OK', True)
    with pytest.raises(RuntimeError, match='still alive'):
        session.interrupt_worker('old', object())


def test_unverified_process_is_never_interrupted():
    session = module.DaskEvaluationSession(1, DaskExecutionConfig())
    with pytest.raises(RuntimeError, match='unverified'):
        session.interrupt_worker('unknown', object())


def test_pid_reuse_is_not_an_alive_old_worker(monkeypatch):
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: SimpleNamespace(
        is_running=lambda: True, create_time=lambda: 20.))
    assert not module._worker_process_alive((123, 10.))


def test_zombie_process_is_not_a_running_fit(monkeypatch):
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: SimpleNamespace(
        is_running=lambda: True, create_time=lambda: 10., status=lambda: module.psutil.STATUS_ZOMBIE))
    assert not module._worker_process_alive((123, 10.))


def test_active_fits_are_interrupted_concurrently_at_total_deadline():
    barrier = Barrier(3)
    released = []

    def interrupt(worker, future):
        barrier.wait(timeout=3)
    flights = [SimpleNamespace(worker=str(i), future=SimpleNamespace(release=lambda: released.append(True)))
               for i in range(3)]
    module._cleanup_flights(SimpleNamespace(interrupt_worker=interrupt), flights, interrupt_active=True)
    assert len(released) == 3
