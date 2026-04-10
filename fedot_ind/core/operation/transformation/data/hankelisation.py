from typing import Optional, Tuple

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

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

    def _transform_input(self, input_data: InputData, is_fit_stage: bool) -> OutputData:
        resolved_window_size, resolved_stride = self._resolve_params(input_data)
        transformed = prepare_lagged_table_data(
            input_data=input_data,
            window_size=resolved_window_size,
            stride=resolved_stride,
            is_fit_stage=is_fit_stage,
        )
        output = self._convert_to_output(
            input_data,
            transformed.features,
            data_type=DataTypesEnum.table,
        )
        output.target = transformed.target
        output.data_type = DataTypesEnum.table
        return output

    def transform(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=False)

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        return self._transform_input(input_data, is_fit_stage=True)
