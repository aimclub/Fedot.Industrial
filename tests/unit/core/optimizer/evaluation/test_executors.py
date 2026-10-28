import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os
import random
import time
from pathlib import Path
from threading import Event, Thread

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


def _invalid_result_worker(input_path, result_path):
    import cloudpickle
    Path(result_path).write_bytes(cloudpickle.dumps('not an evaluation result'))


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


@pytest.mark.parametrize('first_seed', [None, 42])
@pytest.mark.parametrize('second_seed', [None, 7])
def test_overlapping_sequential_dispatchers_serialize_rng_scopes(first_seed, second_seed):
    from golem.core.adapter import DirectAdapter
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.graph import OptGraph, OptNode
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
    from fedot_ind.core.optimizer.evaluation.dispatcher import LocalEvaluationDispatcher

    state = random.getstate(), np.random.get_state()
    entered, release, attempted, second_entered = Event(), Event(), Event(), Event()
    draws = [[], []]

    def first(graph):
        draws[0].append(_draw(None)[1:])
        entered.set()
        assert release.wait(10)
        draws[0].append(_draw(None)[1:])
        return SingleObjFitness(1.)

    def second(graph):
        second_entered.set()
        draws[1].extend([_draw(None)[1:], _draw(None)[1:]])
        return SingleObjFitness(1.)

    def run(seed, objective, signal=False):
        dispatcher = LocalEvaluationDispatcher(
            DirectAdapter(), config=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL), random_seed=seed)
        evaluate = dispatcher.dispatch(objective)
        if signal:
            attempted.set()
        result = evaluate([Individual(OptGraph([OptNode('ridge')]))])
        assert len(result) == 1
        return dispatcher.last_plan.tasks[0].seed

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(run, first_seed, first)
            try:
                assert entered.wait(10)
                b = pool.submit(run, second_seed, second, True)
                assert attempted.wait(10)
                assert not second_entered.wait(.1)
            finally:
                release.set()
            seeds = [a.result(timeout=15), b.result(timeout=15)]
        ambient_python, ambient_numpy = random.Random(), np.random.RandomState()
        ambient_python.setstate(state[0])
        ambient_numpy.set_state(state[1])
        for seed, actual in zip(seeds, draws):
            python_rng = ambient_python if seed is None else random.Random(seed)
            numpy_rng = ambient_numpy if seed is None else np.random.RandomState(seed)
            assert actual == [(python_rng.random(), float(numpy_rng.random_sample())) for _ in range(2)]
        assert _draw(None)[1:] == (ambient_python.random(), float(ambient_numpy.random_sample()))
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])


@pytest.mark.parametrize('workers', [1, 3])
def test_long_tiny_population_spool_is_bounded_by_active_workers(monkeypatch, tmp_path, workers):
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import executors

    counts = []

    class CompletedProcess:
        pid = 123
        exitcode = 0

        def __init__(self, target, args):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)
            counts.append(len(list(tmp_path.iterdir())))

        def is_alive(self):
            return False

        def join(self, timeout):
            pass

        def close(self):
            # The files must survive until their owning process has been reaped.
            assert all(Path(path).exists() for path in self.args)

    monkeypatch.setattr(executors.multiprocessing, 'get_context',
                        lambda method: SimpleNamespace(Process=CompletedProcess))
    tasks = tuple(EvaluationTask(str(i), i) for i in range(60))
    results = executors._run_processes(EvaluationPlan(tasks, LocalBackend.PROCESS, workers),
                                       range(60), _draw, CancellationToken(), None, tmp_path)
    assert all(result.status is ExecutionStatus.SUCCEEDED for result in results)
    assert len(results) == 60
    assert max(counts) <= 3 * workers
    assert list(tmp_path.iterdir()) == []


def test_spawn_population_reclaims_spool_before_launching_more_jobs(tmp_path, monkeypatch):
    from fedot_ind.core.optimizer.evaluation import executors

    counts = []
    receive = executors._receive

    def tracked_receive(job):
        counts.append(len(list(tmp_path.iterdir())))
        return receive(job)

    monkeypatch.setattr(executors, '_receive', tracked_receive)
    tasks = tuple(EvaluationTask(str(i), i) for i in range(12))
    plan = EvaluationPlan(tasks, LocalBackend.PROCESS, max_workers=2)
    results = executors._run_processes(plan, range(12), _draw, CancellationToken(), None, tmp_path)
    assert all(result.status is ExecutionStatus.SUCCEEDED for result in results)
    assert len(results) == 12
    assert max(counts) <= 3 * plan.max_workers
    assert list(tmp_path.iterdir()) == []
    assert not multiprocessing.active_children()


@pytest.mark.parametrize('case', [
    'serialization', 'result_serialization', 'invalid_result', 'cancel', 'timeout', 'crash', 'failure'])
def test_terminal_process_jobs_remove_all_owned_files_before_batch_exit(tmp_path, monkeypatch, case):
    from threading import Lock
    from fedot_ind.core.optimizer.evaluation import executors

    task = EvaluationTask('job')
    token = CancellationToken()
    callback, payload, kwargs = _draw, None, {}
    if case == 'serialization':
        payload = Lock()
    elif case == 'result_serialization':
        def callback(payload): return Lock()
    elif case == 'invalid_result':
        monkeypatch.setattr(executors, '_worker', _invalid_result_worker)
    elif case == 'timeout':
        callback, payload = _sleep, 30
        kwargs = {'candidate_timeout_seconds': .3}
        monkeypatch.setattr(executors, '_worker', _partial_worker)
    elif case == 'cancel':
        callback, payload = _sleep, 30
        receive = executors._receive

        def cancel_active(job):
            token.cancel()
            return receive(job)

        monkeypatch.setattr(executors, '_receive', cancel_active)
    elif case == 'crash':
        callback = _crash
    elif case == 'failure':
        callback = _fail
    result, = executors._run_processes(EvaluationPlan((task,), LocalBackend.PROCESS, **kwargs),
                                       [payload], callback, token, None, tmp_path)
    expected = {'cancel': ExecutionStatus.CANCELLED, 'timeout': ExecutionStatus.TIMED_OUT,
                'failure': ExecutionStatus.FAILED}.get(case, ExecutionStatus.WORKER_FAILED)
    assert result.status is expected
    assert list(tmp_path.iterdir()) == []
    assert not multiprocessing.active_children()


def test_cancel_during_serialization_removes_unstarted_input(tmp_path, monkeypatch):
    from fedot_ind.core.optimizer.evaluation import executors

    token = CancellationToken()
    dumps = executors.cloudpickle.dumps

    def serialize_and_cancel(value):
        serialized = dumps(value)
        token.cancel()
        return serialized

    monkeypatch.setattr(executors.cloudpickle, 'dumps', serialize_and_cancel)
    plan = EvaluationPlan((EvaluationTask('unstarted'),), LocalBackend.PROCESS)
    result, = executors._run_processes(plan, [None], _draw, token, None, tmp_path)
    assert result.status is ExecutionStatus.CANCELLED
    assert list(tmp_path.iterdir()) == []


def test_worker_start_failure_closes_process_before_removing_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import executors

    closed = []

    class UnstartedProcess:
        pid = None

        def __init__(self, target, args):
            self.input_path = Path(args[0])

        def start(self):
            raise OSError('cannot start worker')

        def close(self):
            assert self.input_path.exists()
            closed.append(True)

    monkeypatch.setattr(executors.multiprocessing, 'get_context',
                        lambda method: SimpleNamespace(Process=UnstartedProcess))
    plan = EvaluationPlan((EvaluationTask('unstarted'),), LocalBackend.PROCESS)
    result, = executors._run_processes(plan, [None], _draw, CancellationToken(), None, tmp_path)
    assert result.status is ExecutionStatus.WORKER_FAILED
    assert closed == [True]
    assert list(tmp_path.iterdir()) == []
