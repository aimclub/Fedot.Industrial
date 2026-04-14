from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

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
from fedot_ind.core.models.kernel.okhs_runtime import build_okhs_stage_diagnostics
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import run_forecasting_stage_tuning_on_series


@dataclass
class OKHSFDMDForecaster:
    forecast_horizon: int
    q: float = 0.7
    n_modes: int = 5
    window_size: int = 20
    q_policy: str = 'fixed'
    window_policy: str = 'adaptive_cycle_aware'
    trajectory_sampling_policy: str = 'dense'
    trajectory_rank_policy: str = 'explained_dispersion'
    trajectory_rank_value: int | None = None
    trajectory_representation_policy: str = 'projected'
    latent_trajectory_stride_policy: str = 'adaptive'
    latent_trajectory_stride: int | None = None
    mode_selection_policy: str = 'energy'
    mode_energy_threshold: float = 0.95
    prediction_mode_selection_policy: str = 'adaptive_tail_energy'
    max_prediction_modes: int | None = None
    min_prediction_modes: int = 4
    boundary_alignment_policy: str = 'tapered_offset'
    boundary_alignment_decay: float = 4.0
    prediction_stability_threshold: float | None = 0.03
    anti_smoothing_policy: str = 'residual_bridge'
    anti_smoothing_tail_window: int | None = None
    anti_smoothing_amplitude_ratio: float = 0.35
    anti_smoothing_monotone_ratio: float = 0.9
    anti_smoothing_oscillation_floor: float = 0.25
    anti_smoothing_decay: float = 2.5
    anti_smoothing_target_amplitude_ratio: float = 0.8
    device: str = 'cpu'

    def _build_inner_model(self) -> OKHSForecaster:
        return OKHSForecaster(
            q=self.q,
            forecast_horizon=self.forecast_horizon,
            n_modes=self.n_modes,
            method='dmd',
            q_policy=self.q_policy,
            window_policy=self.window_policy,
            trajectory_sampling_policy=self.trajectory_sampling_policy,
            trajectory_rank_policy=self.trajectory_rank_policy,
            trajectory_rank_value=self.trajectory_rank_value,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
            mode_selection_policy=self.mode_selection_policy,
            mode_energy_threshold=self.mode_energy_threshold,
            prediction_mode_selection_policy=self.prediction_mode_selection_policy,
            max_prediction_modes=self.max_prediction_modes,
            min_prediction_modes=self.min_prediction_modes,
            boundary_alignment_policy=self.boundary_alignment_policy,
            boundary_alignment_decay=self.boundary_alignment_decay,
            prediction_stability_threshold=self.prediction_stability_threshold,
            anti_smoothing_policy=self.anti_smoothing_policy,
            anti_smoothing_tail_window=self.anti_smoothing_tail_window,
            anti_smoothing_amplitude_ratio=self.anti_smoothing_amplitude_ratio,
            anti_smoothing_monotone_ratio=self.anti_smoothing_monotone_ratio,
            anti_smoothing_oscillation_floor=self.anti_smoothing_oscillation_floor,
            anti_smoothing_decay=self.anti_smoothing_decay,
            anti_smoothing_target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
            device=self.device,
        )

    def _refresh_diagnostics(self):
        optimization_info = dict(self.inner_model_.get_optimization_info())
        self.raw_diagnostics_ = optimization_info
        self.stage_diagnostics_ = build_okhs_stage_diagnostics(optimization_info)
        self.diagnostics_ = {
            'model_family': 'operator_model',
            'model_name': 'okhs_fdmd_forecaster',
            **self.stage_diagnostics_,
            'raw_okhs': optimization_info,
        }

    def fit(self, time_series: np.ndarray) -> 'OKHSFDMDForecaster':
        series = np.asarray(time_series, dtype=float).reshape(-1)
        self.inner_model_ = self._build_inner_model()
        self.inner_model_.fit(series, window_size=self.window_size)
        self.training_history_ = series
        self._refresh_diagnostics()
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'OKHSFDMDForecaster was trained for horizon={self.forecast_horizon}, got requested horizon={horizon}.'
            )
        source_series = self.training_history_ if time_series is None else np.asarray(time_series, dtype=float).reshape(
            -1)
        forecast = self.inner_model_.predict(source_series)
        if hasattr(forecast, 'cpu'):
            forecast = forecast.cpu()
        values = np.asarray(forecast, dtype=float).reshape(-1)[:horizon]
        self._refresh_diagnostics()
        self.last_prediction_diagnostics_ = {
            'forecast_shape': tuple(int(value) for value in values.shape),
            'first_prediction_value': float(values[0]) if len(values) else None,
        }
        return values

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }


class OKHSFDMDForecasterImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.model_: OKHSFDMDForecaster | None = None

    def fit(self, input_data: InputData):
        self.model_ = OKHSFDMDForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            q=float(self.params.get('q', 0.7)),
            n_modes=int(self.params.get('n_modes', 5)),
            window_size=int(self.params.get('window_size', 20)),
            q_policy=str(self.params.get('q_policy', 'fixed')),
            window_policy=str(self.params.get('window_policy', 'adaptive_cycle_aware')),
            trajectory_sampling_policy=str(self.params.get('trajectory_sampling_policy', 'dense')),
            trajectory_rank_policy=str(self.params.get('trajectory_rank_policy', 'explained_dispersion')),
            trajectory_rank_value=self.params.get('trajectory_rank_value'),
            trajectory_representation_policy=str(self.params.get('trajectory_representation_policy', 'projected')),
            latent_trajectory_stride_policy=str(self.params.get('latent_trajectory_stride_policy', 'adaptive')),
            latent_trajectory_stride=self.params.get('latent_trajectory_stride'),
            mode_selection_policy=str(self.params.get('mode_selection_policy', 'energy')),
            mode_energy_threshold=float(self.params.get('mode_energy_threshold', 0.95)),
            prediction_mode_selection_policy=str(
                self.params.get('prediction_mode_selection_policy', 'adaptive_tail_energy')
            ),
            max_prediction_modes=self.params.get('max_prediction_modes'),
            min_prediction_modes=int(self.params.get('min_prediction_modes', 4)),
            boundary_alignment_policy=str(self.params.get('boundary_alignment_policy', 'tapered_offset')),
            boundary_alignment_decay=float(self.params.get('boundary_alignment_decay', 4.0)),
            prediction_stability_threshold=self.params.get('prediction_stability_threshold', 0.03),
            anti_smoothing_policy=str(self.params.get('anti_smoothing_policy', 'residual_bridge')),
            anti_smoothing_tail_window=self.params.get('anti_smoothing_tail_window'),
            anti_smoothing_amplitude_ratio=float(self.params.get('anti_smoothing_amplitude_ratio', 0.35)),
            anti_smoothing_monotone_ratio=float(self.params.get('anti_smoothing_monotone_ratio', 0.9)),
            anti_smoothing_oscillation_floor=float(self.params.get('anti_smoothing_oscillation_floor', 0.25)),
            anti_smoothing_decay=float(self.params.get('anti_smoothing_decay', 2.5)),
            anti_smoothing_target_amplitude_ratio=float(
                self.params.get('anti_smoothing_target_amplitude_ratio', 0.8)
            ),
            device=str(self.params.get('device', 'cpu')),
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
        return self.predict(input_data)

    def get_diagnostics(self) -> dict[str, Any]:
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'okhs_fdmd_forecaster',
            dict(self.params),
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'okhs_fdmd_forecaster',
                dict(self.params),
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return build_forecasting_stage_tuning_execution(
            'okhs_fdmd_forecaster',
            base_params=dict(self.params),
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning(self, objective, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return run_sequential_stage_tuning(
            'okhs_fdmd_forecaster',
            objective=objective,
            base_params=dict(self.params),
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning_on_series(
            self,
            time_series: np.ndarray,
            *,
            forecast_horizon: int,
            metric_name: str = 'rmse',
            split_spec=None,
            seasonal_period: int = 1,
            stage_updates: dict[str, object] | None = None,
            max_values_per_parameter: int = 3,
            max_stage_candidates: int = 16,
    ) -> dict[str, object]:
        return run_forecasting_stage_tuning_on_series(
            'okhs_fdmd_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params=dict(self.params),
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
