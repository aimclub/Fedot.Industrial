"""Temporal, serialization and scoring laws for the standalone package."""

import json

import numpy as np
import pytest
from hypothesis import given, strategies as st

from industrial_forecasting import (
    Covariate, CovariateAvailability, ForecastScenario, GridError, GridErrorCode,
    ModelScope, OwaStepScaling, PointForecast, QuantileForecast, TimeGrid, pinball_loss, score_forecast,
)


@given(
    st.lists(st.integers(-10_000, 10_000), min_size=1, max_size=24),
    st.integers(1, 24),
    st.sampled_from(tuple(ModelScope)),
)
def test_scenario_transport_preserves_axes_values_and_policy(history, horizon, scope):
    grid = TimeGrid.positions(len(history), horizon)
    feature = Covariate("calendar", "known_future", history, [0] * horizon)
    original = ForecastScenario(grid, history, scope=scope, covariates=(feature,))
    actual = ForecastScenario.from_dict(json.loads(json.dumps(original.to_dict())))
    assert actual.to_dict() == original.to_dict()
    assert actual.horizon == horizon
    assert actual.covariates[0].availability is CovariateAvailability.KNOWN_FUTURE


def test_calendar_grid_snapshot_does_not_alias_inputs():
    history = np.array(["2026-01-31", "2026-02-28"], dtype="datetime64[D]")
    grid = TimeGrid(history, np.array(["2026-03-31", "2026-04-30"], dtype="datetime64[D]"))
    history[:] = "2000-01-01"
    restored = TimeGrid.from_dict(json.loads(json.dumps(grid.to_dict())))
    assert restored.to_dict() == {
        "kind": "datetime", "history": ["2026-01-31", "2026-02-28"], "future": ["2026-03-31", "2026-04-30"],
    }
    assert not restored.history.flags.writeable and not restored.future.flags.writeable


@pytest.mark.parametrize(("history", "future", "error"), [
    ([0, 1], [1, 2], GridErrorCode.OVERLAP),
    ([0, 0], [1, 2], GridErrorCode.ORDER),
    ([0, 2], [5, 4], GridErrorCode.ORDER),
    ([0, np.nan], [1, 2], GridErrorCode.ORDER),
    ([0, 1], [], GridErrorCode.EMPTY),
    ([0, 1], np.array(["2026-04-30"], dtype="datetime64[D]"), GridErrorCode.INCOMPARABLE),
])
def test_grid_rejects_invalid_forecast_coordinates(history, future, error):
    with pytest.raises(GridError) as info:
        TimeGrid(history, future)
    assert info.value.code is error


def test_scenario_accepts_only_known_future_covariates_on_the_forecast_grid():
    with pytest.raises(ValueError, match="cannot provide future"):
        Covariate("observed", "observed", [1, 2], [3, 4])
    with pytest.raises(ValueError, match="requires future"):
        Covariate("calendar", "known_future", [1, 2])
    observed = Covariate("observed", "observed", [1, 2])
    scenario = ForecastScenario(TimeGrid.positions(2, 2), [1, 2], covariates=(observed,))
    assert scenario.covariates[0].future is None
    known = Covariate("calendar", "known_future", [1, 2], [3])
    with pytest.raises(ValueError, match="does not cover the forecast grid"):
        ForecastScenario(scenario.grid, [1, 2], covariates=(known,))
    with pytest.raises(ValueError, match="distinct named"):
        ForecastScenario(scenario.grid, [1, 2], covariates=(observed, observed))


def test_point_and_quantile_forecasts_are_tied_to_the_scenario_axes():
    scenario = ForecastScenario(TimeGrid.positions(3, 2), [[0, 1, 2], [2, 3, 4]], ("A", "B"))
    point = PointForecast.for_scenario(scenario, [[3, 4], [5, 6]])
    assert point.series_ids == ("A", "B")
    np.testing.assert_array_equal(point.future_index, [3, 4])
    with pytest.raises(ValueError, match="match the series identifiers"):
        PointForecast.for_scenario(scenario, [3, 4])
    predictions = np.stack((point.values - 2, point.values + 2), axis=-1)
    quantiles = QuantileForecast.for_scenario(scenario, predictions, (0.25, 0.75))
    np.testing.assert_allclose(pinball_loss(point.values, quantiles), [0.5, 0.5])
    with pytest.raises(ValueError, match="must not cross"):
        QuantileForecast.for_scenario(scenario, predictions[..., ::-1], (0.25, 0.75))
    with pytest.raises(ValueError, match="strictly increasing"):
        QuantileForecast.for_scenario(scenario, predictions, (0.25, 0.25))


def test_forecasting_metrics_preserve_both_owa_policies():
    actual, predicted, train = [6, 8, 5], [7, 6, 8], [3, 7, 3, 7, 3]
    kwargs = dict(history=train, seasonal_period=2)
    tuning = score_forecast("owa", actual, predicted, **kwargs)
    report = score_forecast("owa", actual, predicted, owa_step_scaling=OwaStepScaling.POINTWISE, **kwargs)
    assert tuning.value == pytest.approx(0.6976806584333255)
    assert report.value == tuning.value
    np.testing.assert_allclose(tuning.by_step, [0.3527754590984975, 0.681940138325781, 1.0583263772954925])
    np.testing.assert_allclose(report.by_step, [1, 5 / 14, 75 / 52])
    assert not np.isclose(report.by_step.mean(), report.value)


def test_metric_shapes_and_training_scale():
    with pytest.raises(ValueError, match="same length"):
        score_forecast("mae", [1, 2], [1])
    assert score_forecast("rmse", [3, 4], [0, 0]).value == pytest.approx(np.sqrt(12.5))
    assert score_forecast("smape", [0, 0], [0, 0]).value == 0
    assert score_forecast("mase", [1, 3], [0, 3], history=[0, 0, 0]).value == 0.5
