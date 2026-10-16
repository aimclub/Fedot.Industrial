from typing import Optional, Tuple

import numpy as np

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot_ind.core.models.ts_forecasting.forecasting_runtime import (
    TensorDevicePolicy,
    build_hankel_trajectory_transform,
    series_to_forecast_tensor_batch,
)


def normalize_hankelisation_params(time_series_length: int,
                                   forecast_length: int,
                                   window_size: int,
                                   stride: int) -> Tuple[int, int]:
    min_window_size = 2
    if forecast_length < 1 or time_series_length < forecast_length + min_window_size:
        raise ValueError('Hankelisation requires history for at least one window and a positive forecast horizon')
    max_window_size = max(min_window_size, time_series_length - forecast_length)
    normalized_window_size = int(max(min_window_size, min(max_window_size, window_size)))

    max_stride = max(1, normalized_window_size // 2)
    normalized_stride = int(max(1, min(max_stride, stride)))
    return normalized_window_size, normalized_stride


class HankelisationImplementation(DataOperationImplementation):
    """Industrial time-series to table transformation based on Hankel windows."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.window_size = int(self.params.get('window_size', 10))
        self.stride = int(self.params.get('stride', 1))
        self.resolved_window_size_ = None
        self.resolved_stride_ = None

    def fit(self, input_data: InputData):
        history = self._history(input_data)
        self.resolved_window_size_, self.resolved_stride_ = normalize_hankelisation_params(
            time_series_length=len(history),
            forecast_length=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            stride=self.stride,
        )
        return self

    def _resolve_params(self, input_data: InputData) -> Tuple[int, int]:
        if self.resolved_window_size_ is not None and self.resolved_stride_ is not None:
            return self.resolved_window_size_, self.resolved_stride_
        return normalize_hankelisation_params(
            time_series_length=len(self._history(input_data)),
            forecast_length=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            stride=self.stride,
        )

    @staticmethod
    def _history(input_data: InputData):
        values = np.asarray(input_data.features, dtype=float)
        if values.ndim not in (1, 2) or (values.ndim == 2 and 1 not in values.shape):
            raise ValueError('Hankelisation supports exactly one univariate series')
        history = values.reshape(-1)
        if not np.isfinite(history).all():
            raise ValueError('Hankelisation requires finite history')
        return history

    def _transform_input(self, input_data: InputData, is_fit_stage: bool) -> OutputData:
        resolved_window_size, resolved_stride = self._resolve_params(input_data)
        history = self._history(input_data)
        horizon = input_data.task.task_params.forecast_length
        if is_fit_stage:
            batch = series_to_forecast_tensor_batch(
                history, forecast_horizon=horizon,
                device_policy=TensorDevicePolicy(device='cpu', dtype='float64'),
            )
            transformed = build_hankel_trajectory_transform(
                batch, window_size=resolved_window_size, stride=resolved_stride,
            )
            transformed_features = transformed.features.detach().cpu().numpy().copy()
            transformed_target = transformed.target.detach().cpu().numpy().copy()
            starts = np.arange(0, len(history) - resolved_window_size - horizon + 1, resolved_stride)
            coordinates = np.arange(len(history)) if input_data.idx is None else np.asarray(input_data.idx)
            transformed_idx = coordinates[starts + resolved_window_size]
        else:
            if len(history) < resolved_window_size:
                raise ValueError('Prediction history is shorter than the fitted Hankel window')
            # Stride selects training windows, not the end of the forecast history.
            transformed_features = history[-resolved_window_size:].reshape(1, -1).copy()
            transformed_target = None
            coordinates = None if input_data.idx is None else np.asarray(input_data.idx)
            transformed_idx = (coordinates[:1].copy() if coordinates is not None and len(coordinates) == horizon
                               else np.array([len(history)]))

        output = self._convert_to_output(
            input_data,
            transformed_features,
            data_type=DataTypesEnum.table,
        )
        output.features = transformed_features
        output.target = transformed_target
        output.idx = transformed_idx
        output.data_type = DataTypesEnum.table
        return output

    def transform(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=False)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=True)
