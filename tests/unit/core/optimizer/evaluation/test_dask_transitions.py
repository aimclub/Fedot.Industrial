import pytest

from fedot_ind.core.optimizer.evaluation.dask_transitions import FlightAction, flight_action


@pytest.mark.parametrize('ready', [False, True])
@pytest.mark.parametrize('available', [False, True])
def test_expired_budget_wins_over_late_result_and_worker_loss(ready, available):
    assert flight_action(now=12, started=2, timeout=10, result_ready=ready,
                         worker_available=available) is FlightAction.TIMEOUT


@pytest.mark.parametrize('ready,available,expected', [
    (True, False, FlightAction.RECEIVE), (True, True, FlightAction.RECEIVE),
    (False, False, FlightAction.WORKER_LOST), (False, True, FlightAction.WAIT)])
@pytest.mark.parametrize('timeout', [None, 11])
def test_polling_transition_precedence(ready, available, expected, timeout):
    assert flight_action(now=12, started=2, timeout=timeout, result_ready=ready,
                         worker_available=available) is expected


def test_expired_future_is_interrupted_without_polling_it(monkeypatch, tmp_path):
    from itertools import chain, count
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.evaluation import dask_executor as module
    from fedot_ind.core.optimizer.evaluation.contracts import EvaluationTask, ExecutionStatus
    from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
    from fedot_ind.core.optimizer.evaluation.executors import CancellationToken

    interrupted, released = [], []

    class Future:
        def done(self):
            pytest.fail('expired future must not be polled')

        def release(self):
            released.append(True)

    future = Future()
    clock = chain((0, 0), count(1))
    monkeypatch.setattr(module, 'monotonic', lambda: next(clock))
    session = SimpleNamespace(
        client=SimpleNamespace(submit=lambda *a, **k: future), digest='one', worker_addresses={'one'},
        available_workers=lambda busy: list({'one'} - set(busy)),
        interrupt_worker=lambda *a: interrupted.append(True))
    plan = DaskEvaluationPlan((EvaluationTask('candidate'),), candidate_timeout_seconds=1)
    result, = module._run(session, plan, [1], abs, CancellationToken(), None,
                          DaskExecutionConfig(), None, tmp_path)
    assert result.status is ExecutionStatus.TIMED_OUT
    assert interrupted == [True] and released == [True]
