import pytest

from benchmark.industrial.evaluation.search_phases import summarize_search_phases


@pytest.mark.parametrize('lifecycle,other', [('search', 2), ('batch', 3)])
def test_phase_accounting_does_not_double_count_session_cleanup(lifecycle, other):
    result = summarize_search_phases(setup_seconds=2, search_seconds=10,
                                     batches=[dict(preparation_seconds=1, execution_seconds=5, application_seconds=1)],
                                     sessions=[dict(startup_seconds=2, cleanup_seconds=1)], lifecycle=lifecycle,
                                     refit_seconds=4, predict_seconds=.1)
    assert result['coordinator_other_seconds'] == other
    assert result['session_startup_seconds'] == 2
    assert result['refit_seconds'] == 4


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), -1])
def test_invalid_phase_measurements_are_rejected(seconds):
    with pytest.raises(ValueError):
        summarize_search_phases(setup_seconds=seconds, search_seconds=1, batches=[], sessions=[],
                                lifecycle='search', refit_seconds=None, predict_seconds=None)
