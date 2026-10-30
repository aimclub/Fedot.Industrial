"""A slow worker restart may not hold up a neighbour's on-time durable result."""

from pathlib import Path
from threading import Event
from time import sleep
from types import SimpleNamespace

import cloudpickle
import pytest

from fedot_ind.core.optimizer.evaluation import dask_executor as module
from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask, ExecutionResult, ExecutionStatus
from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
from fedot_ind.core.optimizer.evaluation.executors import CancellationToken


def test_restart_does_not_block_receiving_other_candidates(tmp_path, monkeypatch):
    clock = [0.]
    restarting, received = Event(), Event()
    interrupted = []
    paths = {}

    class Future:
        status = 'pending'

        def __init__(self, name):
            self.name = name

        def done(self):
            return self.name == 'good' and restarting.is_set()

        def release(self):
            if self.name == 'good':
                received.set()

    def submit(function, task, payload, evaluate, digest, result_path, **kwargs):
        paths[task.candidate_id] = Path(result_path)
        if task.candidate_id == 'slow':
            clock[0] = .5  # Neighbour is submitted later and still has budget.
        return Future(task.candidate_id)

    def interrupt(worker, future):
        interrupted.append(worker)
        result = ExecutionResult('good', ExecutionStatus.SUCCEEDED, .1, value=17)
        paths['good'].write_bytes(cloudpickle.dumps(result))
        restarting.set()
        if not received.wait(2):
            raise RuntimeError('Coordinator was blocked on the worker restart')

    def tick(delay):
        clock[0] += .02
        sleep(.001)

    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'sleep', tick)
    session = SimpleNamespace(
        client=SimpleNamespace(submit=submit), digest='one', worker_addresses={'a', 'b'},
        available_workers=lambda busy: sorted({'a', 'b'} - set(busy)), interrupt_worker=interrupt)
    tasks = (EvaluationTask('slow'), EvaluationTask('good'))
    results = module._run(session, DaskEvaluationPlan(tasks, 2, 1), [0, 0], abs,
                          CancellationToken(), None, DaskExecutionConfig(), None, tmp_path)
    assert results[0].status is ExecutionStatus.TIMED_OUT
    assert results[1].status is ExecutionStatus.SUCCEEDED and results[1].value == 17
    assert interrupted == ['a'] and received.is_set()


def test_async_interruption_failure_is_fatal_and_joined():
    def broken(*args):
        raise RuntimeError('worker is still alive')

    session = SimpleNamespace(interrupt_worker=broken)
    with pytest.raises(module.DaskCleanupError, match='asynchronous'):
        with module._WorkerInterruptions(session, 1) as interruptions:
            interruptions.submit(SimpleNamespace(worker='one', future=None))
    assert all(not thread.is_alive() for thread in interruptions.pool._threads)
