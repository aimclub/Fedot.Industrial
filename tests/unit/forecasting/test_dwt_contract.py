import numpy as np
import pytest

from industrial_forecasting.dwt import ArinoSpec, arino_components, predict_arino, right_edge_dwt

pytest.importorskip("pywt")


def test_forecast_padding_is_removed_and_branches_recover_odd_prefix():
    train = np.arange(89.) + 2 * np.sin(np.arange(89.) / 3)
    spec = ArinoSpec(padding=7)
    slow, seasonal = arino_components(train, spec)
    np.testing.assert_allclose(slow + seasonal, train, atol=1e-12)
    result = predict_arino(
        train, 3, spec,
        trend_forecast=lambda values, h: np.repeat(values[-1], h),
        seasonal_forecast=lambda values, h: np.repeat(values[-1], h),
    )
    np.testing.assert_allclose(result, np.repeat(train[-1], 3))


@pytest.mark.parametrize("step", [2, 4])
def test_block_availability_and_prefix_invariance(step):
    train = np.sin(np.arange(110.))
    changed = train.copy()
    changed[92:] += 100
    first = right_edge_dwt(train, first_end=84, block_size=step)
    second = right_edge_dwt(changed, first_end=84, block_size=step)
    for a, b in zip(first, second):
        assert a.available_at - a.start + 1 == step
        np.testing.assert_allclose(a.bands.sum(axis=1), train[a.start:a.available_at + 1], atol=1e-12)
        if a.available_at < 92:
            np.testing.assert_array_equal(a.bands, b.bands)


def test_mismatched_branch_forecast_is_rejected():
    with pytest.raises(ValueError, match="exactly"):
        predict_arino(np.arange(84.), 2, trend_forecast=lambda *_: np.ones(3))
