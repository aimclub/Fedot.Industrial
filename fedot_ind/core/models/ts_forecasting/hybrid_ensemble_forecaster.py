from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

try:  # pragma: no cover - benchmark/lightweight envs may not have fedot installed
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot.core.repository.dataset_types import DataTypesEnum
except Exception:  # pragma: no cover
    InputData = OutputData = None


    class ModelImplementation:  # type: ignore[override]
        def __init__(self, params=None):
            self.params = params or {}

        def _convert_to_output(self, input_data, predict=None, data_type=None):
            return type('OutputData', (),
                        {'predict': predict, 'data_type': data_type, 'idx': getattr(input_data, 'idx', None)})


    class OperationParameters(dict):  # type: ignore[override]
        def get(self, key, default=None):
            return super().get(key, default)


    class DataTypesEnum:  # pragma: no cover - only used in full FEDOT runtime
        table = 'table'

from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    ForecastingSplitSpec,
    TensorDevicePolicy,
    WeightedAverageHead,
    evaluate_forecast,
    series_to_forecast_tensor_batch,
    split_forecasting_batch,
)
from fedot_ind.core.models.ts_forecasting.havok_forecaster import HAVOKForecaster
from fedot_ind.core.models.ts_forecasting.lagged_ridge_forecaster import LaggedRidgeForecaster
from fedot_ind.core.models.ts_forecasting.low_rank_lagged_ridge_forecaster import LowRankLaggedRidgeForecaster
from fedot_ind.core.models.ts_forecasting.stage_tuning import build_forecasting_stage_tuning_plan


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
    forecast_horizon: int
    complex_branch: str = 'okhs'
    calibration_horizon: int | None = None
    device: str = 'cpu'
    lagged_params: dict[str, Any] = field(default_factory=dict)
    low_rank_params: dict[str, Any] = field(default_factory=dict)
    complex_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.device_policy_ = TensorDevicePolicy(device=self.device)

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
        return OKHSForecaster(
            forecast_horizon=self.forecast_horizon,
            **self.complex_params,
        )

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
            except Exception as exc:  # pragma: no cover - fallback is deterministic but branch failures are rare
                fallback = np.full(requested_horizon, calibration_series[-1], dtype=float)
                branch_forecasts.append(fallback)
                branch_diagnostics[branch_name] = {'status': 'fallback', 'message': str(exc)}
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
        series = np.asarray(time_series, dtype=float).reshape(-1)
        minimum_length = max(self.forecast_horizon * 3, self.forecast_horizon + 12)
        if len(series) > minimum_length:
            try:
                self.branch_calibration_ = self._fit_calibration_head(series)
            except Exception as exc:  # pragma: no cover - deterministic fallback for short/noisy histories
                self.branch_calibration_ = {
                    'calibration_horizon': self.forecast_horizon,
                    'branch_diagnostics': {'status': 'fallback', 'message': str(exc)},
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
        for model in self.branch_models_.values():
            model.fit(series)
        if not hasattr(self, 'ensemble_head_'):
            self.ensemble_head_ = WeightedAverageHead(device_policy=self.device_policy_)
            self.ensemble_head_.weights_ = torch.as_tensor(
                self.branch_calibration_['ensemble_head']['weights'],
                dtype=torch.float32,
            )
        self.diagnostics_ = {
            'model_family': 'hybrid_ensemble',
            'branch_names': tuple(self.branch_models_.keys()),
            'branch_calibration': self.branch_calibration_,
            'ensemble_head': self.ensemble_head_.get_diagnostics(),
        }
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'HybridEnsembleForecaster was trained for horizon={self.forecast_horizon}, got requested horizon={horizon}.'
            )
        source_series = self.training_history_ if time_series is None else np.asarray(time_series, dtype=float).reshape(
            -1)
        branch_predictions = {}
        for branch_name, model in self.branch_models_.items():
            forecast, _ = _safe_forecast(model, source_series, horizon)
            branch_predictions[branch_name] = forecast
        stacked = torch.as_tensor(
            np.vstack([branch_predictions[name] for name in self.branch_models_.keys()]),
            dtype=torch.float32,
        )
        ensemble_forecast = self.ensemble_head_.predict(stacked).detach().cpu().numpy().reshape(-1)
        self.last_prediction_diagnostics_ = {
            'branch_predictions': {name: values.tolist() for name, values in branch_predictions.items()},
            'ensemble_weights': self.ensemble_head_.get_diagnostics().get('weights', []),
        }
        return ensemble_forecast[:horizon]

    def get_diagnostics(self) -> dict[str, object]:
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }


class HybridEnsembleForecasterImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.complex_branch = str(self.params.get('complex_branch', 'okhs'))
        self.calibration_horizon = self.params.get('calibration_horizon')
        self.device = str(self.params.get('device', 'cpu'))
        self.lagged_params = dict(self.params.get('lagged_params', {}))
        self.low_rank_params = dict(self.params.get('low_rank_params', {}))
        self.complex_params = dict(self.params.get('complex_params', {}))
        self.model_: HybridEnsembleForecaster | None = None

    def fit(self, input_data: InputData):
        self.model_ = HybridEnsembleForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            complex_branch=self.complex_branch,
            calibration_horizon=self.calibration_horizon,
            device=self.device,
            lagged_params=self.lagged_params,
            low_rank_params=self.low_rank_params,
            complex_params=self.complex_params,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.predict(np.asarray(input_data.features, dtype=float)), dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'hybrid_ensemble_forecaster',
            {
                'complex_branch': self.complex_branch,
                'calibration_horizon': self.calibration_horizon,
                'lagged_params': self.lagged_params,
                'low_rank_params': self.low_rank_params,
                'complex_params': self.complex_params,
            },
        ).to_dict()
