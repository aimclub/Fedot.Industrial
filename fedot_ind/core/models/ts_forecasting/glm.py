from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - lightweight envs may miss FEDOT runtime
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


class GLMIndustrial(ModelImplementation):
    """Compatibility forecasting stub kept only to preserve repository imports after module reorganisation."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params or OperationParameters())
        self.strategy_ = str(self.params.get('strategy', 'mean'))
        self.forecast_horizon_ = None
        self.last_value_ = 0.0
        self.mean_value_ = 0.0

    def fit(self, input_data: InputData):
        values = np.asarray(input_data.features, dtype=float).reshape(-1)
        self.forecast_horizon_ = int(input_data.task.task_params.forecast_length)
        self.last_value_ = float(values[-1]) if values.size else 0.0
        self.mean_value_ = float(np.mean(values)) if values.size else 0.0
        return self

    def predict(self, input_data: InputData) -> OutputData:
        horizon = int(self.forecast_horizon_ or input_data.task.task_params.forecast_length)
        if self.strategy_ == 'last':
            prediction = np.full(horizon, self.last_value_, dtype=float)
        else:
            prediction = np.full(horizon, self.mean_value_, dtype=float)
        return self._convert_to_output(input_data, predict=prediction, data_type=DataTypesEnum.table)

    def predict_for_fit(self, input_data: InputData):
        return self.predict(input_data)
