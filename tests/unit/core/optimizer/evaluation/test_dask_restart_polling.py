"""A slow worker restart may not hold up a neighbour's on-time durable result."""

from functools import partial
from threading import Event
from types import SimpleNamespace

import pytest

from fedot_ind.core.optimizer.evaluation import dask_bootstrap
from fedot_ind.core.optimizer.evaluation import dask_executor as module
from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask, ExecutionStatus
from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
from fedot_ind.core.optimizer.evaluation.executors import CancellationToken


def test_restart_does_not_block_receiving_other_candidates(tmp_path, monkeypatch):
    clock = [0.]
    published, received = Event(), Event()
    interrupted = []
    writes = {}

    class Future:
        status = 'pending'

        def __init__(self, name):
            self.name = name

        def done(self):
            return self.name == 'good' and published.is_set()

        def release(self):
            if self.name == 'good':
                received.set()

    def submit(function, task, payload, evaluate, digest, result_path, **kwargs):
        writes[task.candidate_id] = partial(function, task, payload, evaluate, digest, result_path)
        if task.candidate_id == 'slow':
            clock[0] = .5  # Neighbour is submitted later and still has budget.
        return Future(task.candidate_id)

    def interrupt(worker, future):
        interrupted.append(worker)
        writes['good']()
        published.set()
        if not received.wait(2):
            raise RuntimeError('Coordinator was blocked on the worker restart')

    def tick(delay):
        if clock[0] < 1.:
            clock[0] = 1.  # Only the first candidate has reached its deadline.
        elif not published.wait(2):
            raise RuntimeError('The neighbour did not publish a result during restart')

    monkeypatch.setattr(dask_bootstrap, 'get_worker', lambda: SimpleNamespace(_industrial_evaluation_manifest='one'))
    monkeypatch.setattr(module, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module, 'sleep', tick)
    session = SimpleNamespace(
        client=SimpleNamespace(submit=submit), digest='one', worker_addresses={'a', 'b'},
        available_workers=lambda busy: sorted({'a', 'b'} - set(busy)), interrupt_worker=interrupt)
    tasks = (EvaluationTask('slow'), EvaluationTask('good'))
    results = module._run(session, DaskEvaluationPlan(tasks, 2, 1), [0, -17], abs,
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
