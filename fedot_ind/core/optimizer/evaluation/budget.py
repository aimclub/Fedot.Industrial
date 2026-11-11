"""Search termination is a control transition, not another failed offspring."""
from enum import Enum


class SearchBudgetExhausted(RuntimeError):
    """A search with completed evaluations has no time for another batch."""


class ConservativeFrontierExhausted(RuntimeError):
    """No locked family has an unsubmitted neighbour in the final phase."""


class SearchStopReason(str, Enum):
    STOP_CRITERIA = 'stop_criteria'
    DEADLINE = 'deadline'
    FRONTIER_EXHAUSTED = 'frontier_exhausted'


def search_deadline_reached(timeout_seconds, elapsed_seconds):
    return timeout_seconds is not None and elapsed_seconds >= timeout_seconds
