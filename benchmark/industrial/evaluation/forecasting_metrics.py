"""Task-local forecasting metric calculations."""
from __future__ import annotations

import math
import numpy as np
from benchmark.industrial.errors import BenchmarkConfigurationError


def _seasonal_naive_forecast(train: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    base = train[-lag:]
    repeats = int(math.ceil(horizon / lag))
    return np.tile(base, repeats)[:horizon]


def _mase_scale(train: np.ndarray, seasonal_period: int) -> float:
    lag = seasonal_period if seasonal_period > 1 and len(train) > seasonal_period else 1
    if len(train) <= lag:
        return 1.0
    scale = np.mean(np.abs(train[lag:] - train[:-lag]))
    return float(scale if scale > 1e-8 else 1.0)


def compute_forecasting_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> float:
    """Compute an aggregate forecasting metric for one horizon vector."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise BenchmarkConfigurationError('Metric inputs must have the same length.')
    if metric_name == 'mae':
        return float(np.mean(np.abs(actual - predicted)))
    if metric_name == 'rmse':
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return float(100.0 * np.mean(2.0 * np.abs(predicted - actual) / denominator))
    if metric_name == 'mase':
        return float(np.mean(np.abs(actual - predicted)) / _mase_scale(train, seasonal_period))
    if metric_name == 'owa':
        baseline = _seasonal_naive_forecast(train, len(actual), seasonal_period)
        smape = compute_forecasting_metric('smape', actual, predicted, train, seasonal_period)
        mase = compute_forecasting_metric('mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_forecasting_metric('smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_forecasting_metric('mase', actual, baseline, train, seasonal_period)
        baseline_smape = baseline_smape if baseline_smape > 1e-8 else 1.0
        baseline_mase = baseline_mase if baseline_mase > 1e-8 else 1.0
        return float(0.5 * ((smape / baseline_smape) + (mase / baseline_mase)))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')


def compute_pointwise_metric(
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        seasonal_period: int,
) -> np.ndarray:
    """Compute horizon-wise metric values for publication artifacts."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if metric_name == 'mae':
        return np.abs(actual - predicted)
    if metric_name == 'rmse':
        return np.sqrt((actual - predicted) ** 2)
    if metric_name == 'smape':
        denominator = np.abs(actual) + np.abs(predicted)
        denominator = np.where(denominator == 0, 1e-8, denominator)
        return 100.0 * 2.0 * np.abs(predicted - actual) / denominator
    if metric_name == 'mase':
        return np.abs(actual - predicted) / _mase_scale(np.asarray(y_train, dtype=float), seasonal_period)
    if metric_name == 'owa':
        train = np.asarray(y_train, dtype=float)
        baseline = _seasonal_naive_forecast(train, len(actual), seasonal_period)
        pointwise_smape = compute_pointwise_metric('smape', actual, predicted, train, seasonal_period)
        pointwise_mase = compute_pointwise_metric('mase', actual, predicted, train, seasonal_period)
        baseline_smape = compute_pointwise_metric('smape', actual, baseline, train, seasonal_period)
        baseline_mase = compute_pointwise_metric('mase', actual, baseline, train, seasonal_period)
        baseline_smape = np.where(baseline_smape <= 1e-8, 1.0, baseline_smape)
        baseline_mase = np.where(baseline_mase <= 1e-8, 1.0, baseline_mase)
        return 0.5 * ((pointwise_smape / baseline_smape) + (pointwise_mase / baseline_mase))
    raise BenchmarkConfigurationError(f'Unsupported metric: {metric_name}')
