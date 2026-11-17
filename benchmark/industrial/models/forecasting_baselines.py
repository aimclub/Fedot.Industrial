"""Task-local forecasting baseline models."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
from benchmark.industrial.core import ForecastingModelAdapter, ForecastingSeriesRecord, RunStatus
from industrial_forecasting.baselines import BaselineKind, forecast_baseline


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
        return forecast_baseline(BaselineKind.LAST_VALUE, series_record.train_values, series_record.forecast_horizon)


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
        return forecast_baseline(BaselineKind.MEAN, series_record.train_values, series_record.forecast_horizon)


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
        return forecast_baseline(BaselineKind.DRIFT, series_record.train_values, series_record.forecast_horizon)


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
        return forecast_baseline(
            BaselineKind.MOVING_AVERAGE, series_record.train_values, series_record.forecast_horizon,
            window_size=max(self.window_size, 1),
        )


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
        return forecast_baseline(BaselineKind.LINEAR_TREND, series_record.train_values, series_record.forecast_horizon)
