from datetime import timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.evaluation.budget import SearchBudgetExhausted, search_deadline_reached
from tests.unit.core.optimizer.evaluation.test_dispatcher import _dispatcher, _individual, _size


@given(st.floats(min_value=0, max_value=1e8, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0, max_value=1e8, allow_nan=False, allow_infinity=False))
def test_deadline_is_monotonic_in_elapsed_time(first, second):
    early, late = sorted((first, second))
    assert not search_deadline_reached(None, late)
    if search_deadline_reached(600, early):
        assert search_deadline_reached(600, late)


def test_no_reproduction_retries_or_preparation_after_search_deadline():
    timer = SimpleNamespace(timeout=timedelta(seconds=600), seconds_from_start=0.)
    dispatcher = _dispatcher()
    evaluate = dispatcher.dispatch(_size, timer)
    assert evaluate([_individual()])
    evaluations = dispatcher.diagnostics_recorder.snapshot().summary
    timer.seconds_from_start = 600.
    for _ in range(4):
        with pytest.raises(SearchBudgetExhausted):
            evaluate([_individual()])
    assert dispatcher.diagnostics_recorder.snapshot().summary == evaluations
    assert dispatcher.evaluation_cache == {}


def test_deadline_exception_does_not_escape_initial_no_result_failure_path():
    timer = SimpleNamespace(timeout=timedelta(seconds=600), seconds_from_start=601.)
    dispatcher = _dispatcher()
    evaluate = dispatcher.dispatch(_size, timer)
    assert evaluate([_individual()]) == []
    assert dispatcher.last_results[0].error.error_type == 'EvaluationTimeout'


def test_new_dispatch_resets_completed_search_state():
    dispatcher = _dispatcher()
    assert dispatcher.dispatch(_size)([_individual()])
    timer = SimpleNamespace(timeout=timedelta(seconds=600), seconds_from_start=601.)
    assert dispatcher.dispatch(_size, timer)([_individual()]) == []
