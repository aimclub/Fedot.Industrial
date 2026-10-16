import multiprocessing
import os
import random
import time
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.evaluation import (
    CancellationToken, EvaluationPlan, EvaluationTask, ExecutionStatus, LocalBackend, execute_plan,
)


def _draw(payload):
    return payload, random.random(), float(np.random.random())


def _sleep(seconds):
    time.sleep(seconds)
    return "completed"


def _crash(payload):
    os._exit(13)


def _fail(payload):
    raise ValueError("invalid candidate")


def _signal_then_sleep(path):
    Path(path).write_text('started', encoding='utf-8')
    time.sleep(30)


def _partial_worker(input_path, result_path):
    Path(result_path).with_suffix('.partial').write_bytes(b'incomplete')
    time.sleep(30)


@given(st.lists(st.integers(min_value=0, max_value=1000), unique=True, max_size=20))
def test_plan_conserves_order_count_and_seeded_results(values):
    plan = EvaluationPlan(tuple(EvaluationTask(str(v), v) for v in values))
    before_python, before_numpy = random.getstate(), np.random.get_state()
    results = execute_plan(plan, values, _draw)
    assert [r.candidate_id for r in results] == list(map(str, values))
    assert all(r.status is ExecutionStatus.SUCCEEDED for r in results)
    assert [r.value for r in results] == [
        (v, random.Random(v).random(), float(np.random.RandomState(v).random_sample())) for v in values]
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])


@pytest.mark.parametrize("backend", list(LocalBackend))
def test_failure_is_one_explicit_outcome_without_success_value(backend):
    plan = EvaluationPlan((EvaluationTask("bad"),), backend)
    result, = execute_plan(plan, [None], _fail)
    assert result.status is ExecutionStatus.FAILED
    assert result.value is None
    assert result.error.error_type == "ValueError"
    assert result.error.message == "invalid candidate"


def test_process_seed_equivalence_and_order_are_independent_of_completion_order():
    tasks = tuple(EvaluationTask(str(i), 42 + i) for i in range(3))
    serial = execute_plan(EvaluationPlan(tasks), range(3), _draw)
    process = execute_plan(EvaluationPlan(tasks, LocalBackend.PROCESS, max_workers=2), range(3), _draw)
    assert [r.value for r in process] == [r.value for r in serial]
    assert [r.candidate_id for r in process] == ["0", "1", "2"]


@pytest.mark.parametrize("backend", list(LocalBackend))
def test_pre_cancelled_plan_never_calls_callback(backend):
    token = CancellationToken()
    token.cancel()
    tasks = tuple(EvaluationTask(str(i)) for i in range(3))
    results = execute_plan(EvaluationPlan(tasks, backend), [None] * 3, _crash, cancellation=token)
    assert len(results) == 3
    assert all(r.status is ExecutionStatus.CANCELLED for r in results)


def test_serial_deadline_is_cooperative_and_late_value_is_not_accepted():
    plan = EvaluationPlan((EvaluationTask("slow"), EvaluationTask("unstarted")), total_timeout_seconds=0.01)
    results = execute_plan(plan, [0.03, 10], _sleep)
    assert all(r.status is ExecutionStatus.TIMED_OUT for r in results)


def test_process_deadline_kills_fit_and_reaps_children():
    before = {child.pid for child in multiprocessing.active_children()}
    plan = EvaluationPlan((EvaluationTask("slow"),), LocalBackend.PROCESS, candidate_timeout_seconds=0.3)
    started = time.monotonic()
    result, = execute_plan(plan, [30], _sleep)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert time.monotonic() - started < 8
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_process_total_timeout_covers_pending_and_active_candidates():
    tasks = tuple(EvaluationTask(str(i)) for i in range(4))
    results = execute_plan(EvaluationPlan(tasks, LocalBackend.PROCESS, total_timeout_seconds=0.3), [30] * 4, _sleep)
    assert len(results) == 4
    assert all(r.status is ExecutionStatus.TIMED_OUT for r in results)
    assert not multiprocessing.active_children()


def test_cancel_running_processes_does_not_wait_for_long_fits():
    token = CancellationToken()
    canceller = Thread(target=lambda: (time.sleep(0.3), token.cancel()))
    canceller.start()
    try:
        tasks = tuple(EvaluationTask(str(i)) for i in range(3))
        results = execute_plan(EvaluationPlan(tasks, LocalBackend.PROCESS, 2), [30] * 3, _sleep,
                               cancellation=token)
    finally:
        canceller.join()
    assert all(r.status is ExecutionStatus.CANCELLED for r in results)
    assert not multiprocessing.active_children()


def test_crashed_worker_has_an_outcome_and_next_candidate_still_runs():
    result, = execute_plan(EvaluationPlan((EvaluationTask("crash"),), LocalBackend.PROCESS), [None], _crash)
    assert result.status is ExecutionStatus.WORKER_FAILED
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("changes", [
    {"tasks": [EvaluationTask("x")]}, {"tasks": (EvaluationTask("x"), EvaluationTask("x"))},
    {"backend": "threads"}, {"max_workers": True}, {"max_workers": 2},
    {"candidate_timeout_seconds": float("nan")}, {"total_timeout_seconds": 0},
    {"total_timeout_seconds": True},
])
def test_invalid_plans_are_rejected_before_effects(changes):
    with pytest.raises(ValueError):
        EvaluationPlan(**{"tasks": (), **changes})


def test_payload_count_must_match_plan():
    with pytest.raises(ValueError, match="exactly one"):
        execute_plan(EvaluationPlan(()), ["extra"], _crash)


def test_unserializable_payload_is_a_worker_failure_not_a_missing_result():
    from threading import Lock
    result, = execute_plan(EvaluationPlan((EvaluationTask("lock"),), LocalBackend.PROCESS), [Lock()], _draw)
    assert result.status is ExecutionStatus.WORKER_FAILED
    assert result.error.error_type == 'TypeError'
    assert not multiprocessing.active_children()


def test_coordinator_interruption_reaps_all_active_children(monkeypatch):
    from fedot_ind.core.optimizer.evaluation import executors

    def interrupt(job):
        raise KeyboardInterrupt

    monkeypatch.setattr(executors, '_receive', interrupt)
    tasks = (EvaluationTask('a'), EvaluationTask('b'))
    with pytest.raises(KeyboardInterrupt):
        execute_plan(EvaluationPlan(tasks, LocalBackend.PROCESS, 2), [30, 30], _sleep)
    assert not multiprocessing.active_children()


def test_cancellation_interrupts_an_already_started_callback(tmp_path):
    marker = tmp_path / 'started'
    token = CancellationToken()

    def cancel_after_start():
        limit = time.monotonic() + 12
        while not marker.exists() and time.monotonic() < limit:
            time.sleep(0.01)
        token.cancel()

    canceller = Thread(target=cancel_after_start)
    canceller.start()
    try:
        result, = execute_plan(EvaluationPlan((EvaluationTask('running'),), LocalBackend.PROCESS),
                               [str(marker)], _signal_then_sleep, cancellation=token)
    finally:
        canceller.join()
    assert marker.exists(), 'The callback must have actually started before cancellation'
    assert result.status is ExecutionStatus.CANCELLED
    assert not multiprocessing.active_children()


def test_partial_worker_output_never_blocks_deadline_or_leaks_tempfiles(monkeypatch):
    from tempfile import TemporaryDirectory
    from fedot_ind.core.optimizer.evaluation import executors
    owned = []

    def tracked_directory(**kwargs):
        directory = TemporaryDirectory(**kwargs)
        owned.append(Path(directory.name))
        return directory

    monkeypatch.setattr(executors, 'TemporaryDirectory', tracked_directory)
    monkeypatch.setattr(executors, '_worker', _partial_worker)
    plan = EvaluationPlan((EvaluationTask('partial'),), LocalBackend.PROCESS, candidate_timeout_seconds=1)
    result, = execute_plan(plan, [None], _draw)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert not any(path.exists() for path in owned)
    assert not multiprocessing.active_children()


def test_publication_between_stat_and_worker_exit_is_not_lost(tmp_path):
    import cloudpickle
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import executors
    from fedot_ind.core.optimizer.evaluation.contracts import ExecutionResult
    path = tmp_path / 'result'
    expected = ExecutionResult('race', ExecutionStatus.SUCCEEDED, 0.1, value=42)

    def exit_after_publication():
        path.write_bytes(cloudpickle.dumps(expected))
        return False

    job = SimpleNamespace(result_path=path, task=EvaluationTask('race'), started=time.monotonic(),
                          process=SimpleNamespace(is_alive=exit_after_publication))
    assert executors._receive(job) == expected


@pytest.mark.parametrize('stop', ['candidate', 'total', 'cancel'])
def test_late_or_cancelled_result_is_rejected_after_deserialization(monkeypatch, stop):
    from fedot_ind.core.optimizer.evaluation import executors
    actual_clock = time.monotonic
    offset = [0.0]
    token = CancellationToken()
    receive = executors._receive

    def delayed_receive(job):
        result = receive(job)
        if result is not None:
            if stop == 'cancel':
                token.cancel()
            else:
                offset[0] += 100
        return result

    monkeypatch.setattr(executors, 'monotonic', lambda: actual_clock() + offset[0])
    monkeypatch.setattr(executors, '_receive', delayed_receive)
    kwargs = ({'candidate_timeout_seconds': 30} if stop == 'candidate' else
              {'total_timeout_seconds': 30} if stop == 'total' else {})
    result, = execute_plan(EvaluationPlan((EvaluationTask('late'),), LocalBackend.PROCESS, **kwargs),
                           [None], _draw, cancellation=token)
    assert result.status is (ExecutionStatus.CANCELLED if stop == 'cancel' else ExecutionStatus.TIMED_OUT)
    assert result.value is None
    assert not multiprocessing.active_children()


def test_interruption_immediately_after_start_keeps_child_owned(monkeypatch):
    from multiprocessing.process import BaseProcess
    start = BaseProcess.start

    def start_and_interrupt(process):
        start(process)
        raise KeyboardInterrupt

    monkeypatch.setattr(BaseProcess, 'start', start_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_plan(EvaluationPlan((EvaluationTask('startup'),), LocalBackend.PROCESS), [30], _sleep)
    assert not multiprocessing.active_children()
