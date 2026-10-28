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
