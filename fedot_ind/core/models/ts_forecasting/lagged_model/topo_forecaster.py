from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
            return type(
                'OutputData',
                (),
                {'predict': predict, 'data_type': data_type,
                    'idx': getattr(input_data, 'idx', None)},
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

from fedot_ind.core.operation.transformation.data.point_cloud import (
    TopologicalEmbeddingConfig, PointCloudBuilder,
    PersistenceConfig, PersistenceDiagramsExtractor
)
from fedot_ind.core.operation.transformation.representation.topological.topofeatures import (
    BettiNumbersSumFeature,
    HolesNumberFeature,
    MaxHoleLifeTimeFeature,
    PersistenceEntropyFeature,
    TopologicalFeaturesExtractor)


def _default_topological_features() -> dict[str, object]:
    return {
        'holes_num': HolesNumberFeature(max_homology_dim=1),
        'max_lifetime': MaxHoleLifeTimeFeature(max_homology_dim=1),
        'entropy': PersistenceEntropyFeature(max_homology_dim=1),
        'betti_sum': BettiNumbersSumFeature(max_homology_dim=1),
    }


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
    filtration_type: str = 'vietoris-rips'
    homology_dimensions: tuple[int, ...] = (0, 1)
    features_dict: dict[str, object] = field(
        default_factory=_default_topological_features)
    multivariate_strategy: str = 'independent'

    def __post_init__(self):
        resolved_channel_model = str(self.channel_model).lower()
        if resolved_channel_model != 'ridge':
            raise ValueError(
                "TopologicalRidgeForecaster currently supports only channel_model='ridge'.")
        self.device_policy_ = TensorDevicePolicy(
            device=self.device, dtype=self.dtype)
        self.runtime_ = ForecastingRuntimeAdapter(self.device_policy_)

        embed_config = TopologicalEmbeddingConfig(
            window_size=self.patch_len,
            stride=self.stride,
            delay=1,
            multivariate_strategy=self.multivariate_strategy
        )
        self.point_cloud_builder_ = PointCloudBuilder(embed_config)

        dist_device = 'cuda' if 'cuda' in str(
            self.device_policy_.device) else 'cpu'
        pers_config = PersistenceConfig(
            homology_dimensions=self.homology_dimensions,
            filtration_type=self.filtration_type,
            normalize=True,
            distance_device=dist_device
        )
        self.persistence_extractor_ = PersistenceDiagramsExtractor(pers_config)

        self.feature_extractor_ = TopologicalFeaturesExtractor(
            self.features_dict)

    @torch.no_grad()
    def _transform_windows(self, windows: np.ndarray) -> np.ndarray:
        windows_tensor = torch.tensor(
            windows,
            dtype=getattr(torch, self.dtype, torch.float32),
            device=self.device_policy_.device
        )

        point_clouds = self.point_cloud_builder_.build(windows_tensor)

        if self.multivariate_strategy == 'independent':
            B, C, M, W = point_clouds.shape
            point_clouds = point_clouds.reshape(B * C, M, W)

        diagrams = self.persistence_extractor_.transform(point_clouds)

        if isinstance(diagrams, np.ndarray):
            diagrams = torch.tensor(
                diagrams,
                dtype=getattr(torch, self.dtype, torch.float32),
                device=self.device_policy_.device
            )

        feature_tensors, _ = self.feature_extractor_.transform(diagrams)
        features_tensor = torch.cat(feature_tensors, dim=1)

        if self.multivariate_strategy == 'independent':
            features_tensor = features_tensor.view(B, -1)

        return features_tensor.detach().cpu().numpy()

    def _resolve_window_size(self, batch: ForecastTensorBatch) -> int:
        return resolve_window_size(
            series_length=batch.series_length,
            forecast_horizon=self.forecast_horizon,
            window_size=self.window_size,
            window_size_percent=self.window_size_percent,
        )

    def fit(self, time_series: np.ndarray) -> 'TopologicalRidgeForecaster':
        """Fit topological Hankel features and a ridge forecast head."""
        batch = self.runtime_.make_batch(
            time_series, forecast_horizon=self.forecast_horizon)
        self.resolved_window_size_ = self._resolve_window_size(batch)
        self.channel_count_ = int(batch.channel_count)
        self.transform_result_ = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.stride,
        )

        training_windows = self.transform_result_.features.detach().clone()
        if training_windows.ndim == 2:
            training_windows = training_windows.view(
                -1, self.channel_count_, self.resolved_window_size_)

        self.topological_features_ = self._transform_windows(
            training_windows.cpu().numpy())

        self.head_ = RidgeForecastingHead(
            alpha=self.alpha, device_policy=self.device_policy_)
        self.head_.fit(self.topological_features_,
                       self.transform_result_.target)

        self.training_history_ = batch.history.detach().clone()
        self.diagnostics_ = {
            'model_family': 'lagged_linear_topological',
            'trajectory_transform': {
                **self.transform_result_.to_dict(),
                'patch_len': self.patch_len,
                'representation': 'topological_hankel_multivariate',
            },
            'forecast_head': self.head_.get_diagnostics(),
            'runtime': batch.to_dict(),
            'extracted_features': getattr(self, 'last_feature_names_', [])
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
            self.training_history_.detach().cpu().numpy(
            ) if time_series is None else time_series,
            forecast_horizon=self.forecast_horizon,
        )
        latest_transform = build_hankel_trajectory_transform(
            batch,
            window_size=self.resolved_window_size_,
            stride=self.transform_result_.stride,
        )
        latest_windows = latest_transform.latest_features.detach().clone()
        if latest_windows.ndim == 2:
            latest_windows = latest_windows.view(
                -1, self.channel_count_, self.resolved_window_size_)

        latest_features = self._transform_windows(latest_windows.cpu().numpy())

        raw_forecast = self.head_.predict(latest_features)

        forecast_matrix = raw_forecast.reshape(
            self.forecast_horizon, self.channel_count_)
        forecast = forecast_matrix[:horizon, :]

        if self.channel_count_ == 1:
            forecast = forecast.squeeze(1)

        self.last_prediction_diagnostics_ = {
            'latest_window_shape': tuple(int(value) for value in latest_windows.shape),
            'topological_feature_shape': tuple(int(value) for value in latest_features.shape),
            'forecast_shape': tuple(int(value) for value in forecast.shape),
        }

        return forecast.detach().cpu().numpy() if isinstance(forecast, torch.Tensor) else np.asarray(forecast)

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
        self.has_explicit_window_percent_ = 'window_size_percent' in self.params.keys()
        self.window_size_percent = self.params.get('window_size_percent')
        self.patch_len = int(self.params.get('patch_len', 10))
        self.stride = int(self.params.get('stride', 1))
        self.alpha = float(self.params.get('alpha', 1.0))
        self.device = str(self.params.get('device', 'auto'))
        self.model_: TopologicalRidgeForecaster | None = None

        self.filtration_type = str(self.params.get(
            'filtration_type', 'vietoris-rips'))
        self.multivariate_strategy = str(self.params.get(
            'multivariate_strategy', 'independent'))
        self.homology_dimensions = self.params.get(
            'homology_dimensions', (0, 1))

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
            filtration_type=self.filtration_type,
            homology_dimensions=self.homology_dimensions,
            multivariate_strategy=self.multivariate_strategy
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the topological forecast."""
        prediction = self.model_.predict(
            np.asarray(input_data.features, dtype=float))
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
            'filtration_type': self.filtration_type,
            'homology_dimensions': self.homology_dimensions,
            'multivariate_strategy': self.multivariate_strategy
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
            'filtration_type': self.filtration_type,
            'homology_dimensions': self.homology_dimensions,
            'multivariate_strategy': self.multivariate_strategy
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
            'filtration_type': self.filtration_type,
            'homology_dimensions': self.homology_dimensions,
            'multivariate_strategy': self.multivariate_strategy
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
            'filtration_type': self.filtration_type,
            'homology_dimensions': self.homology_dimensions,
            'multivariate_strategy': self.multivariate_strategy
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
                'filtration_type': self.filtration_type,
                'homology_dimensions': self.homology_dimensions,
                'multivariate_strategy': self.multivariate_strategy
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
