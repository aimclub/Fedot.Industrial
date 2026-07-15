from typing import Optional, Tuple

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.repository.industrial_implementations.data_transformation import prepare_lagged_table_data


def normalize_hankelisation_params(time_series_length: int,
                                   forecast_length: int,
                                   window_size: int,
                                   stride: int) -> Tuple[int, int]:
    min_window_size = 2
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
        self.resolved_window_size_, self.resolved_stride_ = normalize_hankelisation_params(
            time_series_length=input_data.features.shape[0],
            forecast_length=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            stride=self.stride,
        )
        return self

    def _resolve_params(self, input_data: InputData) -> Tuple[int, int]:
        if self.resolved_window_size_ is not None and self.resolved_stride_ is not None:
            return self.resolved_window_size_, self.resolved_stride_
        return normalize_hankelisation_params(
            time_series_length=input_data.features.shape[0],
            forecast_length=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            stride=self.stride,
        )

    @staticmethod
    def _aligned_idx(input_data: InputData, row_count: int):
        source_idx = input_data.idx
        if source_idx is None:
            return None

        source_idx = source_idx.reshape(-1)
        if row_count <= source_idx.shape[0]:
            return source_idx[-row_count:]
        return source_idx

    def _transform_input(self, input_data: InputData, is_fit_stage: bool) -> OutputData:
        resolved_window_size, resolved_stride = self._resolve_params(input_data)
        transformed = prepare_lagged_table_data(
            input_data=input_data,
            window_size=resolved_window_size,
            stride=resolved_stride,
            is_fit_stage=is_fit_stage,
        )
        transformed_features = transformed.features
        transformed_target = transformed.target

        # In a forecasting pipeline the downstream regression model should see
        # only the latest lagged state at inference time; otherwise it predicts
        # a batch of overlapping horizon windows that FEDOT can not compare to a
        # single holdout horizon during tuning.
        if not is_fit_stage and input_data.task.task_type is TaskTypesEnum.ts_forecasting:
            transformed_features = HankelMatrix(
                time_series=input_data.features,
                window_size=resolved_window_size,
                strides=resolved_stride,
            ).trajectory_matrix.T[-1:, :]
            if transformed_target is not None and len(transformed_target.shape) > 1:
                transformed_target = transformed_target[-1:, :]

        output = self._convert_to_output(
            input_data,
            transformed_features,
            data_type=DataTypesEnum.table,
        )
        output.features = transformed_features
        output.target = transformed_target
        output.idx = self._aligned_idx(input_data, transformed_features.shape[0])
        output.data_type = DataTypesEnum.table
        return output

    def transform(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=False)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=True)
