"""Classical forecasting models with NumPy as their only runtime dependency."""

from enum import Enum

import numpy as np

from .contracts import ForecastScenario, ModelScope, PointForecast
from .grid import positive_count
from .metrics import seasonal_repeat


class BaselineKind(str, Enum):
    LAST_VALUE = "last_value"
    SEASONAL_NAIVE = "seasonal_naive"
    MEAN = "mean"
    DRIFT = "drift"
    MOVING_AVERAGE = "moving_average"
    LINEAR_TREND = "linear_trend"


def forecast_baseline(
    kind: BaselineKind | str,
    history,
    horizon: int,
    *,
    window_size: int = 3,
    seasonal_period: int = 1,
) -> tuple[np.ndarray, dict[str, str | float | int]]:
    """Return a univariate forecast and the parameters used to produce it."""
    kind = BaselineKind(kind)
    train = np.asarray(history, dtype=float)
    if train.ndim != 1 or not len(train) or not np.isfinite(train).all():
        raise ValueError("Baseline history must be a nonempty, finite vector.")
    horizon = positive_count(horizon, "horizon")
    if kind is BaselineKind.SEASONAL_NAIVE:
        return seasonal_repeat(train, horizon, seasonal_period), {"seasonal_period": int(seasonal_period)}
    if kind is BaselineKind.LAST_VALUE:
        return np.full(horizon, train[-1], dtype=float), {"strategy": "last_value"}
    if kind is BaselineKind.MEAN:
        return np.full(horizon, np.mean(train), dtype=float), {"strategy": "mean"}
    if kind is BaselineKind.MOVING_AVERAGE:
        window = min(positive_count(window_size, "window_size"), len(train))
        return np.full(horizon, np.mean(train[-window:]), dtype=float), {"window_size": window}
    if kind is BaselineKind.DRIFT:
        if len(train) == 1:
            return np.full(horizon, train[-1], dtype=float), {"strategy": "fallback"}
        slope = (train[-1] - train[0]) / (len(train) - 1)
        steps = np.arange(1, horizon + 1, dtype=float)
        return train[-1] + slope * steps, {"strategy": "drift"}
    if len(train) == 1:
        return np.full(horizon, train[-1], dtype=float), {"strategy": "constant"}
    slope, intercept = np.polyfit(np.arange(len(train), dtype=float), train, deg=1)
    steps = np.arange(len(train), len(train) + horizon, dtype=float)
    return intercept + slope * steps, {"slope": float(slope), "intercept": float(intercept)}


def predict_baseline(
    scenario: ForecastScenario,
    kind: BaselineKind | str,
    *,
    window_size: int = 3,
    seasonal_period: int = 1,
) -> PointForecast:
    """Fit each series separately; these models do not use covariates or shared fitting."""
    if scenario.scope is not ModelScope.LOCAL or scenario.covariates:
        raise ValueError("Baselines support local scenarios without covariates.")
    values = np.stack([
        forecast_baseline(kind, row, scenario.horizon, window_size=window_size, seasonal_period=seasonal_period)[0]
        for row in scenario.history
    ])
    return PointForecast.for_scenario(scenario, values)
