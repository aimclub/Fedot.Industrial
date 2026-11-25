import numpy as np
import pytest
import torch

from industrial_forecasting import ForecastScenario, TimeGrid, score_forecast
from industrial_forecasting.neural.models import build_series_model, forecast_series
from industrial_forecasting.neural.tcn import TCNModule


@pytest.mark.parametrize("name, params, expected", [
    ("lagged_ridge_forecaster", {"alpha": 0.5}, [9.65426636, 9.89682579, 10.18739700]),
    ("low_rank_lagged_ridge_forecaster", {"rank": 3, "rank_truncation_policy": "expert"},
     [9.65411949, 9.89614677, 10.18616867]),
])
def test_native_ridge_preserves_cpu_reference_and_forecast_grid(name, params, expected):
    # Forecasts captured on ee7a3801, before the package was extracted.
    time = np.arange(99, dtype=float)
    series = 0.11 * time + np.sin(time / 4.0)
    dates = np.datetime64("2018-01", "M") + time.astype(int)
    scenario = ForecastScenario(TimeGrid(dates[:-3], dates[-3:]), series[:-3])
    prediction = forecast_series(scenario, name, params={"device": "cpu", "window_size": 16, **params})
    np.testing.assert_array_equal(prediction.future_index, dates[-3:])
    np.testing.assert_allclose(prediction.values[0], expected, rtol=0, atol=2e-5)
    assert score_forecast("mae", series[-3:], prediction.values[0]).value < 0.002


@pytest.mark.parametrize("name", ["lagged_ridge_forecaster"])
def test_ridge_prediction_uses_the_fitted_window_without_reserving_targets(name):
    time = np.arange(60, dtype=float)
    series = np.column_stack((np.sin(time / 3.), 2. * time + np.cos(time / 4.)))
    model = build_series_model(name, forecast_horizon=3, params={"window_size": 16, "stride": 3, "device": "cpu"})
    model.fit(series)
    np.testing.assert_array_equal(model.predict(series[-16:]), model.predict(series))
    with pytest.raises(ValueError, match="at least 16 observations"):
        model.predict(series[-15:])
    with pytest.raises(ValueError, match="requires 2 channels"):
        model.predict(series[-16:, 0])


def test_tcn_module_is_trainable_on_cpu_and_keeps_its_window_shape():
    model = TCNModule(
        input_size=1, input_chunk_length=8, kernel_size=3, num_filters=3, num_layers=2,
        dilation_base=2, weight_norm=False, target_size=1, target_length=2, dropout=0.0, activation="ReLU",
    )
    windows = torch.arange(24, dtype=torch.float32).reshape(3, 8, 1)
    prediction = model(windows)
    assert prediction.shape == (3, 8, 1)
    prediction[:, -2:].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
