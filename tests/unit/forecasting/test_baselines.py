import numpy as np
import pytest

from industrial_forecasting import Covariate, ForecastScenario, ModelScope, TimeGrid
from industrial_forecasting.baselines import BaselineKind, forecast_baseline, predict_baseline
from industrial_forecasting.trajectory import build_hankel, build_page, decode_diagonal_average, decode_page


@pytest.mark.parametrize(("kind", "expected"), [
    (BaselineKind.LAST_VALUE, [7] * 4),
    (BaselineKind.SEASONAL_NAIVE, [5, 7, 5, 7]),
    (BaselineKind.MEAN, [4] * 4),
    (BaselineKind.DRIFT, [9, 11, 13, 15]),
    (BaselineKind.MOVING_AVERAGE, [6] * 4),
    (BaselineKind.LINEAR_TREND, [9, 11, 13, 15]),
])
def test_numpy_baselines_on_fixed_series(kind, expected):
    values, diagnostics = forecast_baseline(kind, [1, 3, 5, 7], 4, window_size=2, seasonal_period=2)
    np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-14)
    assert diagnostics


def test_seasonal_naive_repeats_one_complete_cycle_and_rejects_insufficient_history():
    values, _ = forecast_baseline("seasonal_naive", [7, 8, 9], 5, seasonal_period=3)
    np.testing.assert_array_equal(values, [7, 8, 9, 7, 8])
    with pytest.raises(ValueError, match="full season"):
        forecast_baseline("seasonal_naive", [7, 8], 5, seasonal_period=3)


def test_local_baseline_forecasts_each_series_on_the_same_grid():
    grid = TimeGrid([6, 7, 8], [9, 10])
    history = [[0, 1, 2], [20, 30, 40]]
    scenario = ForecastScenario(grid, history, ("A", "B"))
    prediction = predict_baseline(scenario, "drift")
    assert prediction.series_ids == ("A", "B")
    np.testing.assert_array_equal(prediction.future_index, [9, 10])
    np.testing.assert_array_equal(prediction.values, [[3, 4], [50, 60]])
    with pytest.raises(ValueError, match="local scenarios"):
        predict_baseline(ForecastScenario(grid, history, ("A", "B"), scope=ModelScope.GLOBAL), "drift")
    observed = Covariate("observed", "observed", history)
    with pytest.raises(ValueError, match="without covariates"):
        predict_baseline(ForecastScenario(grid, history, ("A", "B"), covariates=(observed,)), "drift")


def test_embedding_round_trips_the_fixed_series():
    series = np.arange(12, dtype=float)
    hankel = build_hankel(series, window_size=4)
    page = build_page(series, window_size=4)
    np.testing.assert_array_equal(decode_diagonal_average(hankel.matrix), series)
    np.testing.assert_array_equal(decode_page(page.matrix), series)


def test_page_decode_only_reconstructs_complete_nonoverlapping_windows():
    series = np.arange(11, dtype=float)
    page = build_page(series, window_size=4)
    assert page.diagnostics.stride == page.diagnostics.window_size == 4
    np.testing.assert_array_equal(decode_page(page.matrix, original_length=len(series)), series[:8])
    np.testing.assert_array_equal(decode_page(page.matrix, original_length=6), series[:6])

    sliding = build_page(series, window_size=4, stride=2)
    assert sliding.diagnostics.stride == 2
    np.testing.assert_array_equal(sliding.matrix[1], series[2:6])
