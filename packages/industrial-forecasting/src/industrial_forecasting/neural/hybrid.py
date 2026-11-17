"""Standalone HybridEnsembleForecaster implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .runtime import (
    ForecastingSplitSpec,
    TensorDevicePolicy,
    WeightedAverageHead,
    evaluate_forecast,
    series_to_forecast_tensor_batch,
    split_forecasting_batch,
)
from .havok import HAVOKForecaster
from .lagged import LaggedRidgeForecaster
from .low_rank import \
    LowRankLaggedRidgeForecaster


def _safe_forecast(model, series: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        raw_forecast = model.predict(series, forecast_horizon=horizon)
    except TypeError:
        raw_forecast = model.predict(series)
    forecast = np.asarray(raw_forecast, dtype=float).reshape(-1)
    if hasattr(model, 'get_diagnostics'):
        diagnostics = model.get_diagnostics()
    elif hasattr(model, 'get_optimization_info'):
        diagnostics = model.get_optimization_info()
    else:
        diagnostics = {}
    return forecast[:horizon], diagnostics


@dataclass
class HybridEnsembleForecaster:
    """Named composite forecaster that ensembles lagged, low-rank and complex branches."""

    forecast_horizon: int
    complex_branch: str = 'havok'
    branch_failure_policy: str = 'raise'
    calibration_horizon: int | None = None
    device: str = 'auto'
    lagged_params: dict[str, Any] = field(default_factory=dict)
    low_rank_params: dict[str, Any] = field(default_factory=dict)
    complex_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.branch_failure_policy not in {'raise', 'last_value'}:
            raise ValueError(
                "branch_failure_policy must be either 'raise' or 'last_value'."
            )
        self.device_policy_ = TensorDevicePolicy(device=self.device)

    def _recover_branch_failure(
            self,
            *,
            branch_name: str,
            error: Exception,
            source_series: np.ndarray,
            horizon: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the configured policy to a failed branch forecast.

        The ``raise`` policy raises RuntimeError chained from ``error``. The
        ``last_value`` policy returns ``horizon`` copies of the final source value
        and degraded-status diagnostics; it requires a nonempty source series.
        """
        if self.branch_failure_policy == 'raise':
            raise RuntimeError(
                f"Hybrid ensemble branch {branch_name!r} failed."
            ) from error
        return (
            np.full(horizon, source_series[-1], dtype=float),
            {
                'status': 'degraded',
                'policy': self.branch_failure_policy,
                'message': str(error),
                'exception_type': type(error).__name__,
            },
        )

    def _build_branch_models(self):
        return {
            'lagged_linear': LaggedRidgeForecaster(
                forecast_horizon=self.forecast_horizon,
                device=self.device,
                **self.lagged_params,
            ),
            'low_rank_linear': LowRankLaggedRidgeForecaster(
                forecast_horizon=self.forecast_horizon,
                device=self.device,
                **self.low_rank_params,
            ),
            'operator_model': self._build_complex_branch(),
        }

    def _build_complex_branch(self):
        branch_name = str(self.complex_branch).lower()
        if branch_name == 'havok':
            return HAVOKForecaster(
                forecast_horizon=self.forecast_horizon,
                **self.complex_params,
            )
        raise ValueError("Standalone hybrid forecasting supports complex_branch='havok'.")

    def _fit_calibration_head(self, time_series: np.ndarray) -> dict[str, Any]:
        batch = series_to_forecast_tensor_batch(
            time_series,
            forecast_horizon=self.forecast_horizon,
            device_policy=self.device_policy_,
        )
        requested_horizon = int(self.calibration_horizon or self.forecast_horizon)
        split_spec = ForecastingSplitSpec(validation_horizon=requested_horizon)
        train_batch, validation_target = split_forecasting_batch(batch, split_spec)
        calibration_series = train_batch.history.detach().cpu().numpy()
        branch_models = self._build_branch_models()
        branch_forecasts = []
        branch_diagnostics = {}
        branch_metrics = {}
        for branch_name, model in branch_models.items():
            try:
                model.fit(calibration_series)
                forecast, diagnostics = _safe_forecast(model, calibration_series, requested_horizon)
                branch_forecasts.append(forecast)
                branch_diagnostics[branch_name] = diagnostics
                branch_metrics[branch_name] = evaluate_forecast(
                    validation_target.detach().cpu().numpy(),
                    forecast,
                    metric_name='rmse',
                ).to_dict()
            except Exception as exc:
                fallback, diagnostics = self._recover_branch_failure(
                    branch_name=branch_name,
                    error=exc,
                    source_series=calibration_series,
                    horizon=requested_horizon,
                )
                branch_forecasts.append(fallback)
                branch_diagnostics[branch_name] = diagnostics
                branch_metrics[branch_name] = evaluate_forecast(
                    validation_target.detach().cpu().numpy(),
                    fallback,
                    metric_name='rmse',
                ).to_dict()
        forecast_matrix = torch.as_tensor(np.vstack(branch_forecasts), dtype=torch.float32)
        self.ensemble_head_ = WeightedAverageHead(device_policy=self.device_policy_)
        self.ensemble_head_.fit(
            forecast_matrix,
            validation_target.detach().cpu(),
        )
        return {
            'calibration_horizon': requested_horizon,
            'branch_diagnostics': branch_diagnostics,
            'branch_metrics': branch_metrics,
            'ensemble_head': self.ensemble_head_.get_diagnostics(),
        }

    def fit(self, time_series: np.ndarray) -> 'HybridEnsembleForecaster':
        """Fit branch models and calibrate the weighted ensemble head; return self.

        Under the default ``raise`` policy, calibration errors propagate and
        branch fitting errors become RuntimeError with the original cause. The
        ``last_value`` policy records branch failures. Failed calibration or short
        histories use equal fallback weights when no fitted head is already present;
        short histories skip calibration.
        """
        series = np.asarray(time_series, dtype=float).reshape(-1)
        minimum_length = max(self.forecast_horizon * 3, self.forecast_horizon + 12)
        if len(series) > minimum_length:
            try:
                self.branch_calibration_ = self._fit_calibration_head(series)
            except Exception as exc:
                if self.branch_failure_policy == 'raise':
                    raise
                self.branch_calibration_ = {
                    'calibration_horizon': self.forecast_horizon,
                    'branch_diagnostics': {
                        'status': 'degraded',
                        'policy': self.branch_failure_policy,
                        'message': str(exc),
                    },
                    'branch_metrics': {},
                    'ensemble_head': {'weights': [1 / 3, 1 / 3, 1 / 3]},
                }
        else:
            self.branch_calibration_ = {
                'calibration_horizon': self.forecast_horizon,
                'branch_diagnostics': {},
                'branch_metrics': {},
                'ensemble_head': {'weights': [1 / 3, 1 / 3, 1 / 3]},
            }
        self.branch_models_ = self._build_branch_models()
        self.training_history_ = series
        self.branch_fit_failures_ = {}
        for branch_name, model in self.branch_models_.items():
            try:
                model.fit(series)
            except Exception as exc:
                if self.branch_failure_policy == 'raise':
                    raise RuntimeError(
                        f"Hybrid ensemble branch {branch_name!r} failed during fit."
                    ) from exc
                self.branch_fit_failures_[branch_name] = {
                    'status': 'degraded',
                    'policy': self.branch_failure_policy,
                    'message': str(exc),
                    'exception_type': type(exc).__name__,
                }
        if (
                not hasattr(self, 'ensemble_head_')
                or self.ensemble_head_.weights_ is None
        ):
            self.ensemble_head_ = WeightedAverageHead(device_policy=self.device_policy_)
            self.ensemble_head_.weights_ = torch.as_tensor(
                self.branch_calibration_['ensemble_head']['weights'],
                dtype=self.device_policy_.resolve_dtype(),
                device=self.device_policy_.resolve_device(),
            )
        self.diagnostics_ = {
            'model_family': 'hybrid_ensemble',
            'branch_names': tuple(self.branch_models_.keys()),
            'branch_failure_policy': self.branch_failure_policy,
            'branch_fit_failures': self.branch_fit_failures_,
            'branch_calibration': self.branch_calibration_,
            'ensemble_head': self.ensemble_head_.get_diagnostics(),
        }
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        """Forecast with each branch and combine predictions with learned weights.

        Use fitted history when ``time_series`` is None. ``forecast_horizon`` is a
        number of steps; None or zero selects the fitted horizon, and requesting
        more steps raises ValueError. Failed or undersized branch predictions
        raise RuntimeError under the ``raise`` policy, or use the final source
        value under ``last_value``. Store branch forecasts and degradation status
        in ``last_prediction_diagnostics_`` and return the combined forecast.
        """
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'HybridEnsembleForecaster was trained for horizon={self.forecast_horizon}, '
                f'got requested horizon={horizon}.'
            )
        source_series = (self.training_history_ if time_series is None
                         else np.asarray(time_series, dtype=float).reshape(-1))
        branch_predictions = {}
        branch_diagnostics = {}
        for branch_name, model in self.branch_models_.items():
            try:
                forecast, diagnostics = _safe_forecast(model, source_series, horizon)
                if forecast.size < horizon:
                    raise ValueError(
                        f"Branch returned {forecast.size} values for horizon={horizon}."
                    )
                branch_diagnostics[branch_name] = diagnostics
            except Exception as exc:
                forecast, branch_diagnostics[branch_name] = self._recover_branch_failure(
                    branch_name=branch_name,
                    error=exc,
                    source_series=source_series,
                    horizon=horizon,
                )
            branch_predictions[branch_name] = forecast
        stacked = torch.as_tensor(
            np.vstack([branch_predictions[name] for name in self.branch_models_.keys()]),
            dtype=torch.float32,
        )
        ensemble_forecast = self.ensemble_head_.predict(stacked).detach().cpu().numpy().reshape(-1)
        self.last_prediction_diagnostics_ = {
            'branch_predictions': {name: values.tolist() for name, values in branch_predictions.items()},
            'branch_diagnostics': branch_diagnostics,
            'ensemble_weights': self.ensemble_head_.get_diagnostics().get('weights', []),
            'degraded': any(
                diagnostics.get('status') == 'degraded'
                for diagnostics in branch_diagnostics.values()
            ),
        }
        return ensemble_forecast[:horizon]

    def get_diagnostics(self) -> dict[str, object]:
        """Return branch diagnostics, weights and last branch predictions."""
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }
