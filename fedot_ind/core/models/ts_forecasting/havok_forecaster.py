from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

from fedot_ind.core.operation.transformation.data.trajectory_embedding import (
    build_hankel,
    estimate_window,
    truncate_rank,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import build_forecasting_stage_tuning_execution
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import run_forecasting_stage_tuning_on_series


def _intervals_from_mask(mask: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for index, is_active in enumerate(mask.astype(bool)):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            intervals.append((offset + start, offset + index - 1))
            start = None
    if start is not None:
        intervals.append((offset + start, offset + len(mask) - 1))
    return intervals


@dataclass
class HAVOKForecaster:
    forecast_horizon: int
    window_size: int | None = None
    rank: int | None = None
    forcing_threshold_scale: float = 1.0
    forcing_decay: float = 0.85

    def _build_stage_diagnostics(self) -> dict[str, object]:
        return {
            'trajectory_transform': {
                'kind': 'hankel',
                'window_size': int(self.window_size_),
                'stride': 1,
                'forecast_horizon': int(self.forecast_horizon),
                'channel_count': 1,
                'features_shape': tuple(int(value) for value in self.hankel_shape_),
                'latest_features_shape': (1, int(self.window_size_)),
            },
            'decomposition': {
                'strategy': 'full',
                'projected_shape': tuple(int(value) for value in self.latent_states_.shape),
                'basis_shape': tuple(int(value) for value in self.basis_.shape),
                'input_shape': tuple(int(value) for value in self.hankel_shape_),
            },
            'rank_truncation': {
                'policy': 'explained_variance',
                'selected_rank': int(self.selected_rank_),
                'explained_variance_retained': float(self.explained_variance_retained_),
                'projected_shape': tuple(int(value) for value in self.latent_states_.shape),
                'basis_shape': tuple(int(value) for value in self.basis_.shape),
            },
            'forecast_head': {
                'head_type': 'havok_head',
                'state_dimension': int(max(1, self.selected_rank_ - 1)),
                'forcing_threshold_scale': float(self.forcing_threshold_scale),
                'forcing_threshold': float(self.forcing_threshold_),
                'forcing_decay': float(self.forcing_decay),
                'forcing_activity_ratio': float(np.mean(self.forcing_mask_)) if len(self.forcing_mask_) else 0.0,
                **getattr(self, 'last_prediction_diagnostics_', {}),
            },
        }

    def fit(self, time_series: np.ndarray) -> 'HAVOKForecaster':
        series = np.asarray(time_series, dtype=float).reshape(-1)
        resolved_window = int(
            self.window_size or estimate_window(
                series_length=len(series),
                forecast_horizon=self.forecast_horizon,
                min_ratio=0.10,
                max_ratio=0.20,
            )
        )
        resolved_window = int(max(self.forecast_horizon + 2, min(resolved_window, len(series) - 1)))
        hankel = build_hankel(series, window_size=resolved_window)
        truncated = truncate_rank(
            hankel.matrix,
            rank=self.rank,
            explained_variance=0.95,
            min_rank=2,
        )
        latent = truncated.projected_states
        if latent.shape[0] < 3:
            raise ValueError('HAVOK requires at least three latent states after embedding.')

        state = latent[:-1, :-1]
        forcing = latent[:-1, -1]
        next_state = latent[1:, :-1]
        design = np.column_stack([np.ones(state.shape[0]), state, forcing.reshape(-1, 1)])
        coefficients = np.linalg.pinv(design) @ next_state

        forcing_target = latent[1:, -1]
        forcing_design = np.column_stack([np.ones(len(forcing)), forcing])
        forcing_coefficients = np.linalg.pinv(forcing_design) @ forcing_target

        forcing_scale = float(np.std(forcing)) if len(forcing) else 0.0
        forcing_threshold = float(max(1e-8, forcing_scale * self.forcing_threshold_scale))
        forcing_mask = np.abs(forcing) >= forcing_threshold
        self.series_ = series
        self.window_size_ = resolved_window
        self.hankel_shape_ = tuple(int(value) for value in hankel.matrix.shape)
        self.selected_rank_ = int(truncated.selected_rank)
        self.basis_ = truncated.basis
        self.latent_states_ = latent
        self.state_intercept_ = coefficients[0]
        self.state_matrix_ = coefficients[1:-1]
        self.forcing_vector_ = coefficients[-1]
        self.forcing_intercept_ = float(forcing_coefficients[0])
        self.forcing_autoreg_ = float(forcing_coefficients[1])
        self.forcing_threshold_ = forcing_threshold
        self.forcing_values_ = forcing
        self.forcing_mask_ = forcing_mask
        self.explained_variance_retained_ = float(truncated.explained_variance_retained)
        self.diagnostics_ = self.get_diagnostics()
        return self

    def _project_latest_window(self, time_series: np.ndarray | None = None) -> np.ndarray:
        series = self.series_ if time_series is None else np.asarray(time_series, dtype=float).reshape(-1)
        latest_window = build_hankel(series, window_size=self.window_size_).matrix[-1]
        return latest_window @ self.basis_

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        horizon = int(forecast_horizon or self.forecast_horizon)
        current_latent = self._project_latest_window(time_series)
        current_state = current_latent[:-1]
        current_forcing = float(current_latent[-1])
        forecast = []
        forecast_forcing = []
        for _ in range(horizon):
            next_state = self.state_intercept_ + current_state @ self.state_matrix_ + current_forcing * self.forcing_vector_
            next_forcing = self.forcing_intercept_ + self.forcing_autoreg_ * current_forcing
            if abs(current_forcing) < self.forcing_threshold_:
                next_forcing *= self.forcing_decay
            next_latent = np.concatenate([np.asarray(next_state, dtype=float).reshape(-1), [float(next_forcing)]])
            decoded_window = next_latent @ self.basis_.T
            forecast.append(float(decoded_window[-1]))
            forecast_forcing.append(float(next_forcing))
            current_state = np.asarray(next_state, dtype=float).reshape(-1)
            current_forcing = float(next_forcing)
        self.last_prediction_diagnostics_ = {
            'forecast_forcing_values': [float(value) for value in forecast_forcing],
            'forecast_forcing_mask': [bool(abs(value) >= self.forcing_threshold_) for value in forecast_forcing],
            'forecast_active_intervals': [
                [int(start), int(stop)]
                for start, stop in _intervals_from_mask(
                    np.asarray([abs(value) >= self.forcing_threshold_ for value in forecast_forcing], dtype=bool),
                    offset=len(self.series_),
                )
            ],
            'forcing_threshold': float(self.forcing_threshold_),
        }
        return np.asarray(forecast, dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        forcing_intervals = [[int(start), int(stop)] for start, stop in _intervals_from_mask(self.forcing_mask_)]
        diagnostics = {
            'model_family': 'operator_model',
            'window_size': int(self.window_size_),
            'selected_rank': int(self.selected_rank_),
            'state_dimension': int(max(1, self.selected_rank_ - 1)),
            'forcing_threshold_scale': float(self.forcing_threshold_scale),
            'forcing_threshold': float(self.forcing_threshold_),
            'forcing_decay': float(self.forcing_decay),
            'forcing_activity_ratio': float(np.mean(self.forcing_mask_)) if len(self.forcing_mask_) else 0.0,
            'forcing_values': [float(value) for value in np.asarray(self.forcing_values_, dtype=float)],
            'forcing_mask': [bool(value) for value in np.asarray(self.forcing_mask_, dtype=bool)],
            'forcing_active_intervals': forcing_intervals,
            'explained_variance_retained': float(self.explained_variance_retained_),
            'basis_shape': tuple(int(value) for value in self.basis_.shape),
            'latent_state_shape': tuple(int(value) for value in self.latent_states_.shape),
        }
        diagnostics.update(self._build_stage_diagnostics())
        diagnostics.update(getattr(self, 'last_prediction_diagnostics_', {}))
        return diagnostics


class HAVOKForecasterImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = self.params.get('window_size')
        self.rank = self.params.get('rank')
        self.forcing_threshold_scale = self.params.get('forcing_threshold_scale', 1.0)
        self.forcing_decay = self.params.get('forcing_decay', 0.85)
        self.model_: HAVOKForecaster | None = None

    def fit(self, input_data: InputData):
        self.model_ = HAVOKForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            rank=self.rank,
            forcing_threshold_scale=self.forcing_threshold_scale,
            forcing_decay=self.forcing_decay,
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
        return np.asarray(self.model_.latent_states_, dtype=float).T

    def get_diagnostics(self) -> dict[str, object]:
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'havok_forecaster',
            {
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
            },
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'havok_forecaster',
                {
                    'window_size': self.window_size,
                    'rank': self.rank,
                    'forcing_threshold_scale': self.forcing_threshold_scale,
                    'forcing_decay': self.forcing_decay,
                },
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return build_forecasting_stage_tuning_execution(
            'havok_forecaster',
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
            },
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
            'havok_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
