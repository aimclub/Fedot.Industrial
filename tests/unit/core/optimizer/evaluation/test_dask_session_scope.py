from contextlib import contextmanager
from dataclasses import replace

import pytest

from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask, ExecutionStatus
from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_dispatcher import _SESSION, _OBSERVER, dask_session_scope
from fedot_ind.core.optimizer.evaluation.dask_executor import DaskEvaluationSession, execute_dask_plan


def square(value):
    return value * value


def test_scope_requires_entered_session_and_restores_context_on_failure():
    config = DaskExecutionConfig()
    with pytest.raises(ValueError):
        with dask_session_scope(DaskEvaluationSession(1, config)):
            pytest.fail('No unentered session')

    def observer(session): return None
    with dask_session_scope(observer=observer):
        assert _OBSERVER.get() is observer and _SESSION.get() is None
        with pytest.raises(RuntimeError):
            with dask_session_scope():
                assert _OBSERVER.get() is None
                raise RuntimeError('nested')
        assert _OBSERVER.get() is observer
    assert _OBSERVER.get() is None and _SESSION.get() is None


def test_two_batches_reuse_caller_session_and_leave_client_running():
    config = DaskExecutionConfig(memory_limit=1024**3)
    observed = []

    @contextmanager
    def observe(session):
        observed.append(session)
        yield

    with DaskEvaluationSession(1, config) as session:
        client, cluster = session.client, session.cluster
        for index in (2, 3):
            plan = DaskEvaluationPlan((EvaluationTask(str(index), 42),), max_workers=1,
                                      candidate_timeout_seconds=20, total_timeout_seconds=30)
            result, = execute_dask_plan(plan, [index], square, session=session, on_session=observe)
            assert result.status is ExecutionStatus.SUCCEEDED and result.value == index**2
            assert client.status == 'running' and session.cluster is cluster
        with pytest.raises(ValueError, match='too few'):
            execute_dask_plan(replace(plan, max_workers=2), [1], square, session=session)
        with pytest.raises(ValueError, match='must match'):
            execute_dask_plan(plan, [1], square, session=session, config=replace(config, retries=1))
    assert observed == [session, session]
    assert client.status == 'closed'


def test_reused_larger_session_obeys_smaller_plan_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import dask_executor as module
    from fedot_ind.core.optimizer.evaluation.contracts import ExecutionResult
    from fedot_ind.core.optimizer.evaluation.executors import CancellationToken
    active, peaks = set(), []

    class Future:
        status = 'finished'

        def __init__(self, key):
            self.key = key

        def done(self):
            return True

        def release(self):
            active.remove(self.key)

    def submit(*args, **kwargs):
        active.add(kwargs['key'])
        peaks.append(len(active))
        return Future(kwargs['key'])

    addresses = {'a', 'b', 'c'}
    session = SimpleNamespace(client=SimpleNamespace(submit=submit), digest='test', worker_addresses=addresses,
                              available_workers=lambda busy: sorted(addresses - set(busy)))
    monkeypatch.setattr(module, '_receive', lambda flight: ExecutionResult(
        flight.task.candidate_id, ExecutionStatus.SUCCEEDED, 0., value=1))
    tasks = tuple(EvaluationTask(str(i)) for i in range(3))
    results = module._run(session, DaskEvaluationPlan(tasks, max_workers=1), range(3), square,
                          CancellationToken(), None, DaskExecutionConfig(), None, tmp_path)
    assert len(results) == 3 and max(peaks) == 1 and not active


def test_release_error_cannot_skip_cleanup_of_remaining_workers(tmp_path):
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import dask_executor as module
    from fedot_ind.core.optimizer.evaluation.executors import CancellationToken
    token = CancellationToken()
    released, interrupted, submitted = [], [], []

    class Future:
        def __init__(self, key):
            self.key = key

        def release(self):
            released.append(self.key)
            if self.key == submitted[0]:
                raise OSError('cannot release')

    def submit(*args, **kwargs):
        submitted.append(kwargs['key'])
        if len(submitted) == 3:
            token.cancel()
        return Future(kwargs['key'])

    addresses = {'a', 'b', 'c'}
    session = SimpleNamespace(client=SimpleNamespace(submit=submit), digest='test', worker_addresses=addresses,
                              available_workers=lambda busy: sorted(addresses - set(busy)),
                              interrupt_worker=lambda address, future: interrupted.append(future.key))
    tasks = tuple(EvaluationTask(str(i)) for i in range(3))
    with pytest.raises(module.DaskCleanupError):
        module._run(session, DaskEvaluationPlan(tasks, max_workers=3), range(3), square,
                    token, None, DaskExecutionConfig(), None, tmp_path, interrupt_active=True)
    assert interrupted == submitted and released == submitted
