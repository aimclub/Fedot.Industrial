import numpy as np
import pytest

from industrial_forecasting.wavelets import causal_haar


def test_causal_haar_uses_past_lags_and_keeps_original_grid():
    result = causal_haar([1, 2, 3, 4, 5], levels=2)
    assert result.start_index == 3
    np.testing.assert_array_equal(result.values, [[2.5, .5, 1.], [3.5, .5, 1.]])


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_components_reconstruct_and_are_independent_of_unseen_tail(levels):
    history = np.r_[np.linspace(-3., 2., 23), np.full(8, 17.)]
    prefix = causal_haar(history, levels)
    extended = causal_haar(np.r_[history, [-100., 200., 0.]], levels)
    assert prefix.start_index == extended.start_index == 2**levels - 1
    np.testing.assert_array_equal(prefix.values, extended.values[:len(prefix.values)])
    np.testing.assert_allclose(prefix.values.sum(axis=1), history[prefix.start_index:], atol=1e-14)
    assert not np.shares_memory(prefix.values, history)
    assert not prefix.values.flags.writeable


@pytest.mark.parametrize("history,levels", [
    ([1., 2., 3.], 2), ([1., np.nan], 1), ([1, 2], True),
    ([[1], [2]], 1), ([1 + 2j, 2 - 2j], 1), ([1, 2], 1000),
])
def test_reject_incomplete_support_or_invalid_input(history, levels):
    with pytest.raises(ValueError):
        causal_haar(history, levels)
