"""Standalone LaggedRidgeForecaster implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .runtime import (
    ForecastTensorBatch,
    ForecastingRuntimeAdapter,
    RidgeForecastingHead,
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    latest_hankel_features,
    resolve_window_size,
)


def resolve_lagged_window_size(time_series_length: int, window_size_percent: float) -> int:
    """Resolve a lagged window from a percent-based configuration."""
    return resolve_window_size(
        series_length=int(time_series_length),
        forecast_horizon=1,
        window_size_percent=float(window_size_percent),
    )


@dataclass
class LaggedRidgeForecaster:
    """Stage-aware lagged autoregression baseline with a ridge forecast head."""

    forecast_horizon: int
    window_size: int | None = None
    window_size_percent: float | None = 10.0
    stride: int = 1
    alpha: float = 1.0
    device: str = 'auto'
    dtype: str = 'float32'

    def __post_init__(self):
        self.device_policy_ = TensorDevicePolicy(device=self.device, dtype=self.dtype)
        self.runtime_ = ForecastingRuntimeAdapter(self.device_policy_)

    def _resolve_window_size(self, batch: ForecastTensorBatch) -> int:
        return resolve_window_size(
            series_length=batch.series_length,
            forecast_horizon=self.forecast_horizon,
            window_size=self.window_size,
            window_size_percent=self.window_size_percent,
        )

    def fit(self, time_series: np.ndarray) -> 'LaggedRidgeForecaster':
        """Fit hankelisation plus ridge head on a univariate or multivariate series."""
        batch = self.runtime_.make_batch(time_series, forecast_horizon=self.forecast_horizon)
        self.resolved_window_size_ = self._resolve_window_size(batch)
        self.transform_result_ = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.stride,
        )
        self.head_ = RidgeForecastingHead(alpha=self.alpha, device_policy=self.device_policy_)
        self.head_.fit(self.transform_result_.features, self.transform_result_.target)
        self.channel_count_ = int(self.transform_result_.channel_count)
        self.training_history_ = batch.history.detach().clone()
        self.diagnostics_ = {
            'model_family': 'lagged_linear',
            'trajectory_transform': self.transform_result_.to_dict(),
            'forecast_head': self.head_.get_diagnostics(),
            'runtime': batch.to_dict(),
        }
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        """Forecast the requested horizon using the latest lagged window."""
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'LaggedRidgeForecaster was trained for horizon={self.forecast_horizon}, got requested horizon={horizon}.'
            )
        batch = self.runtime_.make_batch(
            self.training_history_.detach().cpu().numpy() if time_series is None else time_series,
            forecast_horizon=self.forecast_horizon,
        )
        features = latest_hankel_features(
            batch,
            window_size=self.resolved_window_size_,
            channel_count=self.channel_count_,
        )
        forecast_vector = self.head_.predict(features).reshape(self.forecast_horizon, self.channel_count_)
        forecast = forecast_vector[:horizon, 0]
        self.last_prediction_diagnostics_ = {
            'latest_feature_shape': tuple(int(value) for value in features.shape),
            'forecast_shape': tuple(int(value) for value in forecast_vector.shape),
        }
        return forecast.detach().cpu().numpy()

    def get_diagnostics(self) -> dict[str, object]:
        """Return trajectory, head and last-prediction diagnostics."""
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }
