"""Benchmark-facing adapter for the canonical forecasting metrics."""
from __future__ import annotations

import numpy as np

from benchmark.industrial.errors import BenchmarkConfigurationError
from industrial_forecasting.metrics import ForecastMetric, OwaStepScaling, score_forecast


def _score(metric, actual, predicted, history, period, step_scaling):
    try:
        name = ForecastMetric(metric)
    except ValueError as error:
        raise BenchmarkConfigurationError(f'Unsupported metric: {metric}') from error
    try:
        return score_forecast(
            name, np.asarray(actual, dtype=float).reshape(-1), np.asarray(predicted, dtype=float).reshape(-1),
            history=np.asarray(history, dtype=float).reshape(-1), seasonal_period=max(1, period),
            owa_step_scaling=step_scaling,
        )
    except ValueError as error:
        raise BenchmarkConfigurationError(str(error)) from error


def compute_forecasting_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> float:
    """Compute an aggregate forecasting metric for one horizon vector."""
    return _score(metric_name, y_true, y_pred, y_train, seasonal_period, OwaStepScaling.HORIZON).value


def compute_pointwise_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> np.ndarray:
    """Compute horizon-wise values with the historical report scaling."""
    return _score(metric_name, y_true, y_pred, y_train, seasonal_period, OwaStepScaling.POINTWISE).by_step.copy()
