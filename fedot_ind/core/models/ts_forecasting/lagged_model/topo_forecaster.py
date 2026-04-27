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
            return type(
                'OutputData',
                (),
                {'predict': predict, 'data_type': data_type, 'idx': getattr(input_data, 'idx', None)},
            )


    class OperationParameters(dict):  # type: ignore[override]
        def get(self, key, default=None):
            return super().get(key, default)


    class DataTypesEnum:  # pragma: no cover
        table = 'table'

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    ForecastTensorBatch,
    ForecastingRuntimeAdapter,
    RidgeForecastingHead,
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    resolve_window_size,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime import (
    run_forecasting_stage_tuning_on_series,
)


def _resolve_topological_patch_len(window_size: int, patch_len: int | None) -> int:
    requested = int(patch_len or max(2, window_size // 2))
    return int(max(2, min(requested, max(2, window_size - 1))))


def _extract_topological_window_features(window: np.ndarray, *, patch_len: int, stride: int) -> np.ndarray:
    from fedot_ind.core.operation.transformation.data.point_cloud import TopologicalTransformation

    series = np.asarray(window, dtype=float).reshape(-1)
    resolved_patch_len = _resolve_topological_patch_len(len(series), patch_len)
    transformer = TopologicalTransformation(
        time_series=series,
        max_simplex_dim=1,
        epsilon=3,
        window_length=resolved_patch_len,
        stride=int(max(1, stride)),
    )
    betti = transformer.time_series_rolling_betti_ripser(series)
    return np.asarray(betti, dtype=float).reshape(-1)


@dataclass
class TopologicalRidgeForecaster:
    """Lagged ridge forecaster that uses topological window descriptors."""

    forecast_horizon: int
    window_size: int | None = None
    window_size_percent: float | None = 10.0
    patch_len: int = 10
    stride: int = 1
    alpha: float = 1.0
    channel_model: str = 'ridge'
    device: str = 'auto'
    dtype: str = 'float32'

    def __post_init__(self):
        resolved_channel_model = str(self.channel_model).lower()
        if resolved_channel_model != 'ridge':
            raise ValueError("TopologicalRidgeForecaster currently supports only channel_model='ridge'.")
        self.device_policy_ = TensorDevicePolicy(device=self.device, dtype=self.dtype)
        self.runtime_ = ForecastingRuntimeAdapter(self.device_policy_)

    def _resolve_window_size(self, batch: ForecastTensorBatch) -> int:
        return resolve_window_size(
            series_length=batch.series_length,
            forecast_horizon=self.forecast_horizon,
            window_size=self.window_size,
            window_size_percent=self.window_size_percent,
        )

    def _transform_windows(self, windows: np.ndarray) -> np.ndarray:
        feature_rows = [
            _extract_topological_window_features(
                row,
                patch_len=self.patch_len,
                stride=self.stride,
            )
            for row in np.asarray(windows, dtype=float)
        ]
        return np.asarray(feature_rows, dtype=float)

    def fit(self, time_series: np.ndarray) -> 'TopologicalRidgeForecaster':
        """Fit topological Hankel features and a ridge forecast head."""
        batch = self.runtime_.make_batch(time_series, forecast_horizon=self.forecast_horizon)
        self.resolved_window_size_ = self._resolve_window_size(batch)
        self.transform_result_ = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.stride,
        )
        training_windows = self.transform_result_.features.detach().cpu().numpy()
        self.topological_features_ = self._transform_windows(training_windows)
        self.resolved_patch_len_ = _resolve_topological_patch_len(self.resolved_window_size_, self.patch_len)
        self.head_ = RidgeForecastingHead(alpha=self.alpha, device_policy=self.device_policy_)
        self.head_.fit(self.topological_features_, self.transform_result_.target)
        self.channel_count_ = int(self.transform_result_.channel_count)
        self.training_history_ = batch.history.detach().clone()
        self.diagnostics_ = {
            'model_family': 'lagged_linear',
            'trajectory_transform': {
                **self.transform_result_.to_dict(),
                'patch_len': int(self.resolved_patch_len_),
                'representation': 'topological_hankel',
            },
            'forecast_head': self.head_.get_diagnostics(),
            'runtime': batch.to_dict(),
        }
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        """Forecast from the latest topological lag window."""
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'TopologicalRidgeForecaster was trained for horizon={self.forecast_horizon}, '
                f'got requested horizon={horizon}.'
            )
        batch = self.runtime_.make_batch(
            self.training_history_.detach().cpu().numpy() if time_series is None else time_series,
            forecast_horizon=self.forecast_horizon,
        )
        latest_transform = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.transform_result_.stride,
        )
        latest_features = self._transform_windows(latest_transform.latest_features.detach().cpu().numpy())
        forecast_vector = self.head_.predict(latest_features).reshape(self.forecast_horizon, self.channel_count_)
        forecast = forecast_vector[:horizon, 0]
        self.last_prediction_diagnostics_ = {
            'latest_feature_shape': tuple(int(value) for value in latest_transform.latest_features.shape),
            'topological_feature_shape': tuple(int(value) for value in latest_features.shape),
            'forecast_shape': tuple(int(value) for value in forecast_vector.shape),
        }
        return forecast.detach().cpu().numpy()

    def get_diagnostics(self) -> dict[str, object]:
        """Return topological transform, head and last-prediction diagnostics."""
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }


class TopologicalAR(ModelImplementation):
    """Topological lagged forecaster aligned with the `lagged_ridge_forecaster` runtime contract."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Read topological lagged forecaster parameters from operation params."""
        params = params or OperationParameters()
        super().__init__(params)
        self.channel_model = str(self.params.get('channel_model', 'ridge'))
        self.window_size = self.params.get('window_size')
        self.has_explicit_window_percent_ = 'window_size_percent' in self.params
        self.window_size_percent = self.params.get('window_size_percent')
        self.patch_len = int(self.params.get('patch_len', 10))
        self.stride = int(self.params.get('stride', 1))
        self.alpha = float(self.params.get('alpha', 1.0))
        self.device = str(self.params.get('device', 'auto'))
        self.model_: TopologicalRidgeForecaster | None = None

    def fit(self, input_data: InputData):
        """Fit the wrapped topological ridge forecaster from FEDOT InputData."""
        self.model_ = TopologicalRidgeForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=None if self.has_explicit_window_percent_ else self.window_size,
            window_size_percent=self.window_size_percent if self.has_explicit_window_percent_ else None,
            patch_len=self.patch_len,
            stride=self.stride,
            alpha=self.alpha,
            channel_model=self.channel_model,
            device=self.device,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the topological forecast."""
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        """Return topological features for fit-time compatibility paths."""
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.topological_features_, dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        """Expose diagnostics from the fitted wrapped model."""
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        """Return the stage-aware tuning plan for topo_forecaster."""
        base_params = {
            'channel_model': self.channel_model,
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'alpha': self.alpha,
            'device': self.device,
        }
        return build_forecasting_stage_tuning_plan('topo_forecaster', base_params).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        """Return topological forecaster search-space slices by stage."""
        base_params = {
            'channel_model': self.channel_model,
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'alpha': self.alpha,
            'device': self.device,
        }
        return tuple(
            stage.to_dict()
            for stage in build_forecasting_stage_search_spaces('topo_forecaster', base_params)
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve proposed topological updates into a stage tuning execution."""
        base_params = {
            'channel_model': self.channel_model,
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'alpha': self.alpha,
            'device': self.device,
        }
        return build_forecasting_stage_tuning_execution(
            'topo_forecaster',
            base_params=base_params,
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning(self, objective, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Run sequential stage tuning with an externally supplied objective."""
        base_params = {
            'channel_model': self.channel_model,
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'alpha': self.alpha,
            'device': self.device,
        }
        return run_sequential_stage_tuning(
            'topo_forecaster',
            objective=objective,
            base_params=base_params,
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
        """Run runtime stage tuning for topo_forecaster on a raw series."""
        return run_forecasting_stage_tuning_on_series(
            'topo_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'channel_model': self.channel_model,
                'window_size': self.window_size,
                'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
                'patch_len': self.patch_len,
                'stride': self.stride,
                'alpha': self.alpha,
                'device': self.device,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
