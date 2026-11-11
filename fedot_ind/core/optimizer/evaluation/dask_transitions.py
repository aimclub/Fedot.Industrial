"""Pure polling decisions; no futures, clocks, clients or resource ownership."""
from enum import Enum


class FlightAction(Enum):
    WAIT = 'wait'
    TIMEOUT = 'timeout'
    RECEIVE = 'receive'
    WORKER_LOST = 'worker_lost'


def flight_action(*, now, started, timeout, result_ready, worker_available):
    if timeout is not None and now - started >= timeout:
        return FlightAction.TIMEOUT
    if result_ready:
        return FlightAction.RECEIVE
    if not worker_available:
        return FlightAction.WORKER_LOST
    return FlightAction.WAIT
