from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    SeasonalNaive,
    Naive,
    RandomWalkWithDrift,
    CrostonClassic,
    CrostonOptimized,
    CrostonSBA,
    TSB,
    Theta,
)

from benchmark.v2.core import (
    ForecastingModelAdapter,
    ForecastingSeriesRecord,
    RunStatus,
    ModelFamily,
)


STATSFORECAST_AVAILABLE = True


class ModelExecutionError(RuntimeError):
    def __init__(self, status: RunStatus, message: str,
                 skip_reason: str | None = None,
                 backend_error: str | None = None,
                 dependency_status: dict[str, str] | None = None,
                 api_status: dict[str, str] | None = None,
                 budget_info: dict[str, Any] | None = None,):
        super().__init__(message)
        self.status = status
        self.message = message
        self.skip_reason = skip_reason
        self.backend_error = backend_error
        self.dependency_status = dependency_status
        self.api_status = api_status
        self.budget_info = budget_info


def _safe_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


class BaseStatsForecastAdapter(ForecastingModelAdapter):

    def availability(self) -> tuple[RunStatus, str]:
        if not STATSFORECAST_AVAILABLE:
            return RunStatus.NOT_AVAILABLE, "statsforecast is not installed. Run: pip install statsforecast"
        return RunStatus.SUCCESS, "ready"

    def _to_statsforecast_format(self, series: np.ndarray, series_id: str) -> pd.DataFrame:
        return pd.DataFrame({
            "unique_id": [series_id] * len(series),
            "ds": pd.date_range(start="2000-01-01", periods=len(series), freq="D"),
            "y": series,
        })

    def _get_forecast_values(self, forecast_df: pd.DataFrame, model_instance: Any) -> np.ndarray:

        model_name = model_instance.__class__.__name__

        if model_name in forecast_df.columns:
            return forecast_df[model_name].values

        matching_cols = [col for col in forecast_df.columns if col.startswith(model_name)]
        if matching_cols:
            return forecast_df[matching_cols[0]].values

        numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return forecast_df[numeric_cols[0]].values

        raise KeyError(f"Cannot find forecast column for model {model_name}")

    def _validate_series_length(self, train: np.ndarray, min_length: int = 3) -> bool:
        return len(train) >= min_length

    def _forecast(self, model, series_record: ForecastingSeriesRecord) -> tuple[
        np.ndarray | None, dict[str, Any] | None]:
        try:
            train = _safe_float_array(series_record.train_values)

            if not self._validate_series_length(train):
                return None, {"error": f"Series too short: length={len(train)}"}

            sf = StatsForecast(models=[model], freq="D", n_jobs=1)
            sf.fit(df=self._to_statsforecast_format(train, series_record.series_id))

            try:
                forecast_df = sf.predict(h=series_record.forecast_horizon, level=list(self.prediction_levels))
            except:
                forecast_df = sf.predict(h=series_record.forecast_horizon)

            forecast = self._get_forecast_values(forecast_df, model)

            return forecast, None
        except Exception as e:
            return None, {"error": str(e), "exception_type": type(e).__name__}

@dataclass
class StatsForecastAutoARIMA(BaseStatsForecastAdapter):

    seasonal: bool = True
    stepwise: bool = True
    approximation: bool = False
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    max_P: int = 2
    max_D: int = 1
    max_Q: int = 2
    start_p: int = 0
    start_q: int = 0
    start_P: int = 0
    start_Q: int = 0
    stationary: bool = False

    name: str = "StatsForecastAutoARIMA"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "arima", "forecasting")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon
        seasonal_period = series_record.seasonal_period

        use_seasonal = self.seasonal and seasonal_period > 1

        if use_seasonal:
            season_length = min(seasonal_period, len(train) // 2) if len(train) > 4 else 1
        else:
            season_length = 1

        model = AutoARIMA(
            seasonal=use_seasonal,
            season_length=season_length,
            stepwise=self.stepwise,
            approximation=self.approximation,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            max_P=self.max_P,
            max_D=self.max_D,
            max_Q=self.max_Q,
            start_p=self.start_p,
            start_q=self.start_q,
            start_P=self.start_P,
            start_Q=self.start_Q,
            stationary=self.stationary,
        )

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        metadata = {
            "seasonal_period": seasonal_period,
            "seasonal_enabled": use_seasonal,
            "season_length_used": season_length,
            "model_params": {
                "stepwise": self.stepwise,
            },
            "last_train_value": float(train[-1]) if len(train) > 0 else None,
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastAutoETS(BaseStatsForecastAdapter):

    seasonal: bool = True
    model: str = "ZZZ"
    damped: bool = None
    allow_multiplicative_trend: bool = False

    name: str = "StatsForecastAutoETS"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "ets", "forecasting")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon
        seasonal_period = series_record.seasonal_period

        # Validate season length
        season_length = seasonal_period if self.seasonal and seasonal_period > 1 else 1
        if season_length > 1 and len(train) < season_length * 2:
            season_length = 1

        model = AutoETS(
            season_length=season_length,
            model=self.model,
            damped=self.damped,
        )

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        metadata = {
            "seasonal_period": seasonal_period,
            "seasonal_enabled": self.seasonal and season_length > 1,
            "model_spec": self.model,
            "last_train_value": float(train[-1]) if len(train) > 0 else None,
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastTheta(BaseStatsForecastAdapter):
    seasonal: bool = True
    decomposition_type: str = "multiplicative"

    name: str = "StatsForecastTheta"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "theta", "forecasting")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon
        seasonal_period = series_record.seasonal_period

        effective_decomp_type = self.decomposition_type
        if self.decomposition_type == "multiplicative" and np.any(train <= 0):
            effective_decomp_type = "additive"

        season_length = seasonal_period if self.seasonal and seasonal_period > 1 else 1
        if season_length > 1 and len(train) < season_length * 2:
            season_length = 1

        model = Theta(
            season_length=season_length,
            decomposition_type=effective_decomp_type,
        )

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        metadata = {
            "seasonal_period": seasonal_period,
            "seasonal_enabled": self.seasonal and season_length > 1,
            "decomposition_type": effective_decomp_type,
            "original_decomposition_type": self.decomposition_type,
            "auto_switched": effective_decomp_type != self.decomposition_type,
            "last_train_value": float(train[-1]) if len(train) > 0 else None,
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastSeasonalNaive(BaseStatsForecastAdapter):

    name: str = "StatsForecastSeasonalNaive"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "baseline", "seasonal_naive")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon
        seasonal_period = series_record.seasonal_period

        season_length = seasonal_period
        if len(train) < season_length:
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                f"Insufficient data for seasonal naive: len={len(train)}, season={season_length}"
            )

        model = SeasonalNaive(season_length=season_length)

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        metadata = {
            "seasonal_period": seasonal_period,
            "method": "seasonal_naive",
            "last_train_value": float(train[-1]) if len(train) > 0 else None,
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastNaive(BaseStatsForecastAdapter):

    name: str = "StatsForecastNaive"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "baseline", "naive")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon

        if len(train) == 0:
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                "Empty series"
            )

        model = Naive()
        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        metadata = {
            "method": "naive",
            "last_train_value": float(train[-1]),
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastDrift(BaseStatsForecastAdapter):

    name: str = "StatsForecastDrift"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "baseline", "drift")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon

        if len(train) < 2:
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                "Insufficient data for drift (need at least 2 observations)"
            )

        model = RandomWalkWithDrift()
        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        drift = (train[-1] - train[0]) / max(len(train) - 1, 1) if len(train) > 1 else 0.0

        metadata = {
            "method": "random_walk_with_drift",
            "drift_slope": float(drift),
            "last_train_value": float(train[-1]),
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastCroston(BaseStatsForecastAdapter):
    """StatsForecast Croston method for intermittent demand."""

    version: str = "optimized"

    name: str = "StatsForecastCroston"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "intermittent", "croston")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast intermittent demand using Croston method."""
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon

        if len(train) < 2:
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                "Insufficient data for Croston (need at least 2 observations)"
            )

        if self.version == "classic":
            model = CrostonClassic()
        elif self.version == "sba":
            model = CrostonSBA()
        else:
            model = CrostonOptimized()

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        zero_ratio = np.mean(train == 0)
        nonzero_mean = np.mean(train[train > 0]) if np.any(train > 0) else 0.0
        nonzero_count = np.sum(train > 0)

        metadata = {
            "version": self.version,
            "zero_ratio": float(zero_ratio),
            "nonzero_mean": float(nonzero_mean),
            "nonzero_observations": int(nonzero_count),
            "method": f"croston_{self.version}",
            "last_train_value": float(train[-1]),
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata


@dataclass
class StatsForecastTSB(BaseStatsForecastAdapter):
    """StatsForecast TSB method."""

    alpha: float = 0.1
    beta: float = 0.1

    name: str = "StatsForecastTSB"
    family: ModelFamily = ModelFamily.CLASSICAL_BASELINE
    tags: tuple[str, ...] = ("statsforecast", "intermittent", "tsb")
    optional: bool = False

    def forecast(self, series_record: ForecastingSeriesRecord) -> tuple[np.ndarray, dict[str, Any]]:
        """Forecast intermittent demand using TSB method."""
        train = _safe_float_array(series_record.train_values)
        horizon = series_record.forecast_horizon

        if len(train) < 2:
            raise ModelExecutionError(
                RunStatus.SKIPPED,
                "Insufficient data for TSB (need at least 2 observations)"
            )

        model = TSB(self.alpha, self.beta)

        forecast, error = self._forecast(model, series_record)

        if forecast is None or error is not None:
            raise ModelExecutionError(
                RunStatus.FAILED,
                error.get("error", "Unknown error") if error else "Forecast failed"
            )

        if len(forecast) != horizon:
            raise ModelExecutionError(
                RunStatus.FAILED,
                f"Forecast length {len(forecast)} does not match horizon {horizon}"
            )

        zero_ratio = np.mean(train == 0)
        nonzero_mean = np.mean(train[train > 0]) if np.any(train > 0) else 0.0

        metadata = {
            "alpha": self.alpha,
            "beta": self.beta,
            "zero_ratio": float(zero_ratio),
            "nonzero_mean": float(nonzero_mean),
            "method": "tsb",
            "last_train_value": float(train[-1]),
            "first_prediction_value": float(forecast[0]) if len(forecast) > 0 else None,
        }

        return forecast, metadata
