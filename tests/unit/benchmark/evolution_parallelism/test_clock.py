from time import time

import pytest

from benchmark.industrial.experiments.evolution_parallelism.clock import absolute_search_timer, search_complete


def test_setup_time_is_not_granted_again_at_timer_entry():
    launch = time() - 40
    timer = absolute_search_timer(launch, launch + 100)
    with timer:
        assert 40 <= timer.seconds_from_start < 45
        assert 55 < timer.timeout.total_seconds() - timer.seconds_from_start <= 60
        assert not timer.is_time_limit_reached()
    with timer:
        assert timer.seconds_from_start >= 40


def test_expired_setup_cannot_start_a_new_search_budget():
    launch = time() - 40
    with absolute_search_timer(launch, launch + 20) as timer:
        assert timer.is_time_limit_reached()


def test_invalid_deadline_rejected():
    with pytest.raises(ValueError):
        absolute_search_timer(10, 10)


@pytest.mark.parametrize('evolved,expected', [(14, False), (15, True), (16, True)])
def test_minimum_generation_bar_is_independent_of_protocol(evolved, expected):
    assert search_complete(evolved, 15) is expected
