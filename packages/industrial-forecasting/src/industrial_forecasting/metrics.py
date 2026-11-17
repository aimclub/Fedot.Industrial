"""Point and quantile metrics without model training or benchmark state."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contracts import QuantileForecast
from .grid import positive_count


class ForecastMetric(str, Enum):
    MAE = "mae"
    RMSE = "rmse"
    SMAPE = "smape"
    MASE = "mase"
    OWA = "owa"


class OwaStepScaling(str, Enum):
    HORIZON = "horizon"
    POINTWISE = "pointwise"


@dataclass(frozen=True, eq=False)
class ForecastScore:
    value: float
    by_step: np.ndarray


def _vector(values, field: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a nonempty, finite vector.")
    return vector


def seasonal_repeat(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """Repeat the last complete cycle, without inspecting any future observations."""
    train = _vector(history, "History")
    horizon = positive_count(horizon, "horizon")
    period = positive_count(period, "seasonal_period")
    if len(train) < period:
        raise ValueError("History must contain one full season.")
    return np.resize(train[-period:], horizon)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.abs(actual) + np.abs(predicted)
    return 100.0 * 2.0 * np.abs(predicted - actual) / np.where(denominator == 0, 1e-8, denominator)


def score_forecast(
    metric: ForecastMetric | str,
    actual,
    predicted,
    *,
    history=None,
    seasonal_period: int = 1,
    owa_step_scaling: OwaStepScaling = OwaStepScaling.HORIZON,
) -> ForecastScore:
    """Score one series. MASE/OWA use only the supplied training history for scale."""
    metric = ForecastMetric(metric)
    actual = _vector(actual, "Actual")
    predicted = _vector(predicted, "Prediction")
    if actual.shape != predicted.shape:
        raise ValueError("Metric inputs must have the same length.")
    error = actual - predicted
    if metric is ForecastMetric.MAE:
        by_step = np.abs(error)
        result = np.mean(by_step)
    elif metric is ForecastMetric.RMSE:
        squared = error ** 2
        by_step = np.sqrt(squared)
        result = np.sqrt(np.mean(squared))
    elif metric is ForecastMetric.SMAPE:
        by_step = _smape(actual, predicted)
        result = np.mean(by_step)
    else:
        train = _vector(history, "Training history")
        period = positive_count(seasonal_period, "seasonal_period")
        # Preserve the established scale for short and constant training series.
        lag = period if len(train) > period else 1
        scale = np.mean(np.abs(train[lag:] - train[:-lag])) if len(train) > lag else 1.0
        scale = scale if scale > 1e-8 else 1.0
        mase = np.abs(error) / scale
        if metric is ForecastMetric.MASE:
            by_step = mase
            result = np.mean(np.abs(error)) / scale
        else:
            baseline = seasonal_repeat(train, len(actual), lag)
            smape = _smape(actual, predicted)
            baseline_smape = _smape(actual, baseline)
            baseline_mase = np.abs(actual - baseline) / scale
            smape_scale, mase_scale = np.mean(baseline_smape), np.mean(baseline_mase)
            smape_scale = smape_scale if smape_scale > 1e-8 else 1.0
            mase_scale = mase_scale if mase_scale > 1e-8 else 1.0
            horizon_ratios = 0.5 * (smape / smape_scale + mase / mase_scale)
            result = np.mean(horizon_ratios)
            if OwaStepScaling(owa_step_scaling) is OwaStepScaling.POINTWISE:
                by_step = 0.5 * (smape / np.where(baseline_smape <= 1e-8, 1.0, baseline_smape)
                                 + mase / np.where(baseline_mase <= 1e-8, 1.0, baseline_mase))
            else:
                by_step = horizon_ratios
    by_step.setflags(write=False)
    return ForecastScore(float(result), by_step)


def pinball_loss(actual, forecast: QuantileForecast) -> np.ndarray:
    """Return the mean pinball loss at each quantile, in forecast level order."""
    target = np.asarray(actual, dtype=float)
    if target.ndim == 1:
        target = target[None, :]
    if target.shape != forecast.values.shape[:2] or not np.isfinite(target).all():
        raise ValueError("Actual values must match the series and horizon of the quantile forecast.")
    residual = target[..., None] - forecast.values
    levels = np.asarray(forecast.levels)
    return np.maximum(levels * residual, (levels - 1) * residual).mean(axis=(0, 1))
