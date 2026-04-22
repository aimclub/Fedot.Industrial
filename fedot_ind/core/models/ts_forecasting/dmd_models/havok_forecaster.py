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
from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    MLPForecastingHead,
    RidgeForecastingHead,
    TensorDevicePolicy,
)
from fedot_ind.core.models.ts_forecasting.progress_policy import resolve_forecasting_progress_policy
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime import (
    run_forecasting_stage_tuning_on_series,
)


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
    head_policy: str = 'mlp'
    head_activation: str = 'relu'
    head_depth: int = 2
    head_base_hidden_dim: int = 512
    head_hidden_dim: int | None = None
    head_hidden_layers: int | None = None
    head_epochs: int = 120
    head_learning_rate: float = 1e-3
    device: str = 'auto'
    dtype: str = 'float32'
    progress_policy: dict[str, object] | bool | None = None

    def __post_init__(self):
        self.device_policy_ = TensorDevicePolicy(device=self.device, dtype=self.dtype)
        self.progress_policy_ = resolve_forecasting_progress_policy(self.progress_policy)

    def _resolve_head_depth(self) -> int:
        if self.head_hidden_layers is not None:
            return int(max(1, self.head_hidden_layers))
        return int(max(1, self.head_depth))

    def _resolve_head_base_hidden_dim(self) -> int:
        if self.head_hidden_dim is not None:
            return int(max(1, self.head_hidden_dim))
        return int(max(1, self.head_base_hidden_dim))

    def _prepare_series(self, time_series: np.ndarray) -> tuple[np.ndarray, int]:
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
        return series, resolved_window

    def _build_latent_embedding(self, series: np.ndarray, resolved_window: int):
        hankel = build_hankel(series, window_size=resolved_window)
        truncated = truncate_rank(
            hankel.matrix,
            rank=self.rank,
            explained_variance=0.99,
            min_rank=2,
        )
        latent = truncated.projected_states
        if latent.shape[0] < 3:
            raise ValueError('HAVOK requires at least three latent states after embedding.')
        return hankel, truncated, latent

    def _build_transition_matrices(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = latent[:-1, :-1]
        forcing = latent[:-1, -1]
        next_state = latent[1:, :-1]
        state_design = np.column_stack([state, forcing.reshape(-1, 1)])
        forcing_target = latent[1:, -1].reshape(-1, 1)
        forcing_design = forcing.reshape(-1, 1)
        return state_design, next_state, forcing_design, forcing_target

    def _build_transition_head(self):
        resolved_policy = str(self.head_policy).lower()
        if resolved_policy == 'linear':
            return RidgeForecastingHead(alpha=0.0, device_policy=self.device_policy_)
        return MLPForecastingHead(
            depth=self._resolve_head_depth(),
            base_hidden_dim=self._resolve_head_base_hidden_dim(),
            activation=str(self.head_activation).lower(),
            epochs=self.head_epochs,
            learning_rate=self.head_learning_rate,
            progress_policy=self.progress_policy_,
            device_policy=self.device_policy_,
        )

    def _fit_transition_heads(self, latent: np.ndarray) -> None:
        state_design, next_state, forcing_design, forcing_target = self._build_transition_matrices(latent)
        self.state_design_shape_ = tuple(int(value) for value in state_design.shape)
        self.state_target_shape_ = tuple(int(value) for value in next_state.shape)
        self.forcing_design_shape_ = tuple(int(value) for value in forcing_design.shape)
        self.forcing_target_shape_ = tuple(int(value) for value in forcing_target.shape)
        self.head_policy_ = str(self.head_policy).lower()
        self.state_head_ = self._build_transition_head()
        self.state_head_.fit(state_design, next_state)
        self.forcing_head_ = self._build_transition_head()
        self.forcing_head_.fit(forcing_design, forcing_target)

    def _compute_forcing_threshold(self, forcing: np.ndarray) -> float:
        forcing_scale = float(np.std(forcing)) if len(forcing) else 0.0
        return float(max(1e-8, forcing_scale * self.forcing_threshold_scale))

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
                'head_policy': str(self.head_policy_),
                'head_activation': str(self.head_activation).lower(),
                'head_depth': int(self._resolve_head_depth()),
                'head_base_hidden_dim': int(self._resolve_head_base_hidden_dim()),
                'state_dimension': int(max(1, self.selected_rank_ - 1)),
                'forcing_threshold_scale': float(self.forcing_threshold_scale),
                'forcing_threshold': float(self.forcing_threshold_),
                'forcing_decay': float(self.forcing_decay),
                'forcing_activity_ratio': float(np.mean(self.forcing_mask_)) if len(self.forcing_mask_) else 0.0,
                'state_design_shape': tuple(int(value) for value in self.state_design_shape_),
                'state_target_shape': tuple(int(value) for value in self.state_target_shape_),
                'forcing_design_shape': tuple(int(value) for value in self.forcing_design_shape_),
                'forcing_target_shape': tuple(int(value) for value in self.forcing_target_shape_),
                'state_head_diagnostics': self.state_head_.get_diagnostics(),
                'forcing_head_diagnostics': self.forcing_head_.get_diagnostics(),
                **getattr(self, 'last_prediction_diagnostics_', {}),
            },
        }

    def fit(self, time_series: np.ndarray) -> 'HAVOKForecaster':
        series, resolved_window = self._prepare_series(time_series)
        hankel, truncated, latent = self._build_latent_embedding(series, resolved_window)
        forcing = latent[:-1, -1]
        self._fit_transition_heads(latent)
        forcing_threshold = self._compute_forcing_threshold(forcing)
        forcing_mask = np.abs(forcing) >= forcing_threshold
        self.series_ = series
        self.window_size_ = resolved_window
        self.hankel_shape_ = tuple(int(value) for value in hankel.matrix.shape)
        self.selected_rank_ = int(truncated.selected_rank)
        self.basis_ = truncated.basis
        self.latent_states_ = latent
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
            state_features = np.concatenate(
                [np.asarray(current_state, dtype=float).reshape(-1), [float(current_forcing)]],
            ).reshape(1, -1)
            next_state = self.state_head_.predict(
                state_features,
            )
            if hasattr(next_state, 'detach'):
                next_state = next_state.detach().cpu().numpy()
            next_state = np.asarray(next_state, dtype=float).reshape(-1)
            next_forcing = self.forcing_head_.predict(np.asarray([[float(current_forcing)]], dtype=float))
            if hasattr(next_forcing, 'detach'):
                next_forcing = next_forcing.detach().cpu().numpy()
            next_forcing = float(np.asarray(next_forcing, dtype=float).reshape(-1)[0])
            if abs(current_forcing) < self.forcing_threshold_:
                next_forcing *= self.forcing_decay
            next_latent = np.concatenate([next_state, [float(next_forcing)]])
            decoded_window = next_latent @ self.basis_.T
            forecast.append(float(decoded_window[-1]))
            forecast_forcing.append(float(next_forcing))
            current_state = next_state
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
        self.head_policy = str(self.params.get('head_policy', 'mlp'))
        self.head_activation = str(self.params.get('head_activation', 'relu'))
        self.head_depth = int(self.params.get('head_depth', 2))
        raw_head_base_hidden_dim = self.params.get('head_base_hidden_dim')
        self.head_base_hidden_dim = 512 if raw_head_base_hidden_dim is None else int(raw_head_base_hidden_dim)
        raw_head_hidden_dim = self.params.get('head_hidden_dim')
        self.head_hidden_dim = None if raw_head_hidden_dim is None else int(raw_head_hidden_dim)
        raw_head_hidden_layers = self.params.get('head_hidden_layers')
        self.head_hidden_layers = None if raw_head_hidden_layers is None else int(raw_head_hidden_layers)
        self.head_epochs = int(self.params.get('head_epochs', 120))
        self.head_learning_rate = float(self.params.get('head_learning_rate', 1e-3))
        self.device = str(self.params.get('device', 'auto'))
        self.progress_policy = self.params.get('progress_policy')
        self.model_: HAVOKForecaster | None = None

    def fit(self, input_data: InputData):
        self.model_ = HAVOKForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            rank=self.rank,
            forcing_threshold_scale=self.forcing_threshold_scale,
            forcing_decay=self.forcing_decay,
            head_policy=self.head_policy,
            head_activation=self.head_activation,
            head_depth=self.head_depth,
            head_base_hidden_dim=self.head_base_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            head_hidden_layers=self.head_hidden_layers,
            head_epochs=self.head_epochs,
            head_learning_rate=self.head_learning_rate,
            device=self.device,
            progress_policy=self.progress_policy,
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
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
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
                    'head_policy': self.head_policy,
                    'head_activation': self.head_activation,
                    'head_depth': self.head_depth,
                    'head_base_hidden_dim': self.head_base_hidden_dim,
                    'head_hidden_dim': self.head_hidden_dim,
                    'head_hidden_layers': self.head_hidden_layers,
                    'head_epochs': self.head_epochs,
                    'head_learning_rate': self.head_learning_rate,
                    'device': self.device,
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
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
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
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
