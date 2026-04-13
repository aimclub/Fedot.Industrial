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

from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    ForecastTensorBatch,
    ForecastingRuntimeAdapter,
    RidgeForecastingHead,
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    compute_randomized_svd_decomposition,
    compute_svd_decomposition,
    compute_tensor_decomposition,
    project_features,
    resolve_window_size,
    truncate_decomposition_rank,
)


@dataclass
class LowRankLaggedRidgeForecaster:
    forecast_horizon: int
    window_size: int | None = None
    window_size_percent: float | None = 10.0
    stride: int = 1
    alpha: float = 1.0
    rank: int | None = None
    explained_variance: float = 0.95
    decomposition_strategy: str = 'full'
    rank_truncation_policy: str = 'explained_variance'
    unfolding_strategy: str = 'channels_last'
    device: str = 'cpu'
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

    def _decompose(self, features):
        strategy = str(self.decomposition_strategy).lower()
        if strategy == 'randomized':
            return compute_randomized_svd_decomposition(features)
        if strategy == 'tensor':
            return compute_tensor_decomposition(features, unfolding_strategy=self.unfolding_strategy)
        return compute_svd_decomposition(features, strategy='full')

    def fit(self, time_series: np.ndarray) -> 'LowRankLaggedRidgeForecaster':
        batch = self.runtime_.make_batch(time_series, forecast_horizon=self.forecast_horizon)
        self.resolved_window_size_ = self._resolve_window_size(batch)
        self.transform_result_ = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.stride,
        )
        self.decomposition_result_ = self._decompose(self.transform_result_.features)
        self.rank_result_ = truncate_decomposition_rank(
            self.decomposition_result_,
            rank=self.rank,
            explained_variance=self.explained_variance,
            policy=self.rank_truncation_policy,
            expert_rank=self.rank,
            min_rank=2,
        )
        self.head_ = RidgeForecastingHead(alpha=self.alpha, device_policy=self.device_policy_)
        self.head_.fit(self.rank_result_.projected_features, self.transform_result_.target)
        self.channel_count_ = int(self.transform_result_.channel_count)
        self.training_history_ = batch.history.detach().clone()
        self.diagnostics_ = {
            'model_family': 'low_rank_linear',
            'trajectory_transform': self.transform_result_.to_dict(),
            'decomposition': self.decomposition_result_.to_dict(),
            'rank_truncation': self.rank_result_.to_dict(),
            'forecast_head': self.head_.get_diagnostics(),
            'runtime': batch.to_dict(),
        }
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'LowRankLaggedRidgeForecaster was trained for horizon={self.forecast_horizon}, got requested horizon={horizon}.'
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
        latest_projected = project_features(latest_transform.latest_features, self.rank_result_.basis)
        forecast_vector = self.head_.predict(latest_projected).reshape(self.forecast_horizon, self.channel_count_)
        forecast = forecast_vector[:horizon, 0]
        self.last_prediction_diagnostics_ = {
            'latest_projected_shape': tuple(int(value) for value in latest_projected.shape),
            'forecast_shape': tuple(int(value) for value in forecast_vector.shape),
        }
        return forecast.detach().cpu().numpy()

    def get_diagnostics(self) -> dict[str, object]:
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }


class LowRankLaggedRidgeForecasterImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = self.params.get('window_size')
        self.has_explicit_window_percent_ = 'window_size_percent' in self.params
        self.window_size_percent = self.params.get('window_size_percent')
        self.stride = int(self.params.get('stride', 1))
        self.alpha = float(self.params.get('alpha', 1.0))
        self.rank = self.params.get('rank')
        self.explained_variance = float(self.params.get('explained_variance', 0.95))
        self.decomposition_strategy = str(self.params.get('decomposition_strategy', 'full'))
        self.rank_truncation_policy = str(self.params.get('rank_truncation_policy', 'explained_variance'))
        self.unfolding_strategy = str(self.params.get('unfolding_strategy', 'channels_last'))
        self.device = str(self.params.get('device', 'cpu'))
        self.model_: LowRankLaggedRidgeForecaster | None = None

    def fit(self, input_data: InputData):
        self.model_ = LowRankLaggedRidgeForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=None if self.has_explicit_window_percent_ else self.window_size,
            window_size_percent=self.window_size_percent if self.has_explicit_window_percent_ else None,
            stride=self.stride,
            alpha=self.alpha,
            rank=self.rank,
            explained_variance=self.explained_variance,
            decomposition_strategy=self.decomposition_strategy,
            rank_truncation_policy=self.rank_truncation_policy,
            unfolding_strategy=self.unfolding_strategy,
            device=self.device,
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
        return np.asarray(self.model_.rank_result_.projected_features.detach().cpu().numpy(), dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()
