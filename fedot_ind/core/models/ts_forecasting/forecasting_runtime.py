"""FEDOT boundary conversions for the standalone tensor runtime."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from industrial_forecasting.neural.runtime import (
    ForecastingSplitKind,
    ForecastTensorBatch,
    ForecastingOperationCapability,
    ForecastingSplitSpec,
    ForecastingFoldSplit,
    ForecastingEvaluationResult,
    TrajectoryTransformResult,
    DecompositionResult,
    RankTruncationResult,
    ForecastHeadResult,
    ForecastingRuntimeAdapter,
    RidgeForecastingHead,
    build_scaled_mlp_hidden_dims,
    MLPForecastingHead,
    WeightedAverageHead,
    ensure_tensor_2d,
    series_to_forecast_tensor_batch,
    resolve_window_size,
    resolve_stride,
    build_hankel_trajectory_transform,
    compute_svd_decomposition,
    compute_randomized_svd_decomposition,
    compute_tensor_decomposition,
    truncate_decomposition_rank,
    project_features,
    inverse_project_features,
    iter_forecasting_splits,
    split_forecasting_batch,
    evaluate_forecast,
    capability_to_dict,
    TensorDevicePolicy,
    ForecastingProgressPolicy,
    resolve_forecasting_progress_policy,
)

try:  # pragma: no cover - optional FEDOT runtime in lightweight envs
    from fedot.core.repository.dataset_types import DataTypesEnum
except Exception:  # pragma: no cover
    class DataTypesEnum:  # type: ignore[override]
        table = 'table'
        ts = 'ts'

from fedot_ind.integration.fedot.compatibility import InputData, OutputData

__all__ = [
    'ForecastingSplitKind',
    'ForecastTensorBatch',
    'ForecastingOperationCapability',
    'ForecastingSplitSpec',
    'ForecastingFoldSplit',
    'ForecastingEvaluationResult',
    'TrajectoryTransformResult',
    'DecompositionResult',
    'RankTruncationResult',
    'ForecastHeadResult',
    'ForecastingRuntimeAdapter',
    'RidgeForecastingHead',
    'build_scaled_mlp_hidden_dims',
    'MLPForecastingHead',
    'WeightedAverageHead',
    'ensure_tensor_2d',
    'series_to_forecast_tensor_batch',
    'resolve_window_size',
    'resolve_stride',
    'build_hankel_trajectory_transform',
    'compute_svd_decomposition',
    'compute_randomized_svd_decomposition',
    'compute_tensor_decomposition',
    'truncate_decomposition_rank',
    'project_features',
    'inverse_project_features',
    'iter_forecasting_splits',
    'split_forecasting_batch',
    'evaluate_forecast',
    'capability_to_dict',
    'TensorDevicePolicy',
    'ForecastingProgressPolicy',
    'resolve_forecasting_progress_policy',
    'ForecastingBoundaryAdapter',
]


class ForecastingBoundaryAdapter:
    """Convert FEDOT boundary objects to and from tensor-native runtime payloads."""

    @staticmethod
    def from_input_data(
            input_data: InputData,
            *,
            forecast_horizon: int | None = None,
            device_policy: TensorDevicePolicy | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ForecastTensorBatch:
        """Build a ForecastTensorBatch from FEDOT InputData."""
        resolved_horizon = (
            int(forecast_horizon)
            if forecast_horizon is not None
            else int(input_data.task.task_params.forecast_length)
        )
        return series_to_forecast_tensor_batch(
            input_data.features,
            forecast_horizon=resolved_horizon,
            device_policy=device_policy,
            idx=getattr(input_data, 'idx', None),
            metadata=metadata,
        )

    @staticmethod
    def to_output_data(
            input_data: InputData,
            forecast: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> OutputData:
        """Wrap a forecast vector back into FEDOT OutputData."""
        if OutputData is None:  # pragma: no cover - lightweight test envs
            return type('OutputData', (), {'predict': forecast, 'data_type': DataTypesEnum.table})()
        prediction = np.asarray(
            forecast.detach().cpu().numpy() if isinstance(forecast, torch.Tensor) else forecast,
            dtype=float,
        )
        return OutputData(
            idx=input_data.idx,
            features=input_data.features,
            predict=prediction,
            target=getattr(input_data, 'target', None),
            task=input_data.task,
            data_type=DataTypesEnum.table,
            supplementary_data=input_data.supplementary_data,
        )
