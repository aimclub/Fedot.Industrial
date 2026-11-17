import numpy as np
import pytest

from industrial_forecasting import ForecastScenario, ModelScope, TimeGrid
from industrial_forecasting.wavelet_hfcm import WaveletHFCMSpec, fit_wavelet_hfcm, predict_wavelet_hfcm
from industrial_forecasting.wavelets import causal_haar


def test_matches_an_independent_augmented_ridge_and_recursive_reference():
    history = np.array([0., 1., 3., 6., 10., 5., 2., 4., 6., 8., 2.])
    spec = WaveletHFCMSpec(2, 2, .25)
    normalized = (history - 5.) / 5.
    bands = causal_haar(normalized, spec.levels).values
    inputs = np.array([[1., *bands[k - spec.order:k].ravel()] for k in range(spec.order, len(bands))])
    penalty = np.eye(inputs.shape[1]) * np.sqrt(spec.alpha)
    penalty[0, 0] = 0.
    bound = np.nextafter(1., 0.)
    targets = np.arctanh(np.clip(bands[spec.order:], -bound, bound))
    coefficients = np.linalg.lstsq(np.vstack((inputs, penalty)),
                                   np.vstack((targets, np.zeros((len(penalty), bands.shape[1])))),
                                   rcond=None)[0]
    past = list(bands[-spec.order:])
    for _ in range(5):
        features = np.r_[1., np.ravel(past[-spec.order:])]
        past.append(np.tanh(features @ coefficients))
    expected = 5. + 5. * np.asarray(past[-5:]).sum(axis=1)
    fitted = fit_wavelet_hfcm(history, spec)
    np.testing.assert_allclose(fitted.predict(5), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(fitted.predict(2), fitted.predict(5)[:2])


def test_constant_series_and_fitted_state_do_not_depend_on_input_mutation():
    history = np.full(16, -12.)
    fitted = fit_wavelet_hfcm(history)
    history[:] = 300.
    np.testing.assert_array_equal(fitted.predict(7), np.full(7, -12.))


@pytest.mark.parametrize("dated", [False, True])
def test_scenario_preserves_order_and_regular_forecast_grid(dated):
    time = np.arange(20)
    if dated:
        time = np.datetime64("2026-01-01") + time.astype("timedelta64[D]")
    rows = np.stack([np.full(16, 4.), np.arange(16) % 5])
    scenario = ForecastScenario(TimeGrid(time[:16], time[16:]), rows, ("A", "B"))
    forecast = predict_wavelet_hfcm(scenario)
    assert forecast.series_ids == ("A", "B")
    np.testing.assert_array_equal(forecast.future_index, time[16:])
    np.testing.assert_array_equal(forecast.values[0], [4.] * 4)
    np.testing.assert_array_equal(forecast.values[1], fit_wavelet_hfcm(rows[1]).predict(4))


def test_refuses_irregular_grid_global_fit_and_insufficient_history():
    rows = np.arange(16)
    with pytest.raises(ValueError, match="constant time step"):
        predict_wavelet_hfcm(ForecastScenario(TimeGrid(np.arange(16), [18, 19]), rows))
    with pytest.raises(ValueError, match="local scenarios"):
        predict_wavelet_hfcm(ForecastScenario(TimeGrid(np.arange(16), [16]), rows, scope=ModelScope.GLOBAL))
    with pytest.raises(ValueError, match="lagged training pair"):
        fit_wavelet_hfcm(np.arange(6))


@pytest.mark.parametrize("parameter", [
    {"levels": False}, {"order": 0}, {"alpha": 0.}, {"alpha": np.inf}, {"alpha": True}, {"alpha": "1"},
])
def test_spec_requires_positive_integer_structure_and_finite_ridge(parameter):
    with pytest.raises(ValueError):
        WaveletHFCMSpec(**parameter)
