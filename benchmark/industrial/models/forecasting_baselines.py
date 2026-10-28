"""Task-local forecasting baseline models."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
from benchmark.industrial.core import ForecastingModelAdapter, ForecastingSeriesRecord, RunStatus


@dataclass
class NaiveLastValueModel(ForecastingModelAdapter):
    """Simple baseline that repeats the last observed value."""

    name: str = 'NaiveLastValue'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the last train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'last_value'}


@dataclass
class NaiveMeanModel(ForecastingModelAdapter):
    """Simple baseline that repeats the train mean."""

    name: str = 'NaiveMean'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the mean train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        return np.full(series_record.forecast_horizon, np.mean(train), dtype=float), {'strategy': 'mean'}


@dataclass
class NaiveDriftModel(ForecastingModelAdapter):
    """Simple baseline that extrapolates a linear end-to-end drift."""

    name: str = 'NaiveDrift'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast a drift line from first to last train value."""
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'fallback'}
        slope = (train[-1] - train[0]) / max(len(train) - 1, 1)
        horizon = np.arange(1, series_record.forecast_horizon + 1, dtype=float)
        return train[-1] + slope * horizon, {'strategy': 'drift'}


@dataclass
class MovingAverageModel(ForecastingModelAdapter):
    """Simple baseline that repeats the trailing moving average."""

    window_size: int = 3
    name: str = 'MovingAverage'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast by repeating the last-window mean."""
        train = np.asarray(series_record.train_values, dtype=float)
        window = min(max(self.window_size, 1), len(train))
        return np.full(series_record.forecast_horizon, np.mean(train[-window:]), dtype=float), {'window_size': window}


@dataclass
class LinearTrendModel(ForecastingModelAdapter):
    """Simple baseline that extrapolates a fitted linear trend."""

    name: str = 'LinearTrend'
    tags: tuple[str, ...] = ('baseline', 'forecasting')
    optional: bool = False

    def availability(self) -> tuple[RunStatus, str]:
        """Return readiness for the dependency-free baseline."""
        return RunStatus.SUCCESS, 'ready'

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Fit a linear trend on train values and extrapolate the horizon."""
        train = np.asarray(series_record.train_values, dtype=float)
        if len(train) <= 1:
            return np.full(series_record.forecast_horizon, train[-1], dtype=float), {'strategy': 'constant'}
        index = np.arange(len(train), dtype=float)
        slope, intercept = np.polyfit(index, train, deg=1)
        future_index = np.arange(len(train), len(train) + series_record.forecast_horizon, dtype=float)
        return intercept + slope * future_index, {'slope': float(slope), 'intercept': float(intercept)}
