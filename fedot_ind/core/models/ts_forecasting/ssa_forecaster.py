from __future__ import annotations

from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.models.ts_forecasting.mssa_forecaster import MSSAForecaster


class SSAForecasterImplementation(ModelImplementation):
    """Compatibility SSA wrapper over the shared Page-embedding forecasting backend."""

    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = params.get('window_size')
        self.rank = params.get('rank')
        self.explained_variance = params.get('explained_variance', 0.95)
        self.history_lookback = max(params.get('history_lookback', 0), 0)
        self.mode = params.get('mode', 'one_dimensional')
        self.compatibility_status_ = 'compatibility_wrapper'
        self.model_: MSSAForecaster | None = None

    def _prepare_series(self, input_data: InputData) -> np.ndarray:
        series = np.asarray(input_data.features, dtype=float).reshape(-1)
        if self.history_lookback and series.shape[0] > self.history_lookback:
            return series[-self.history_lookback:]
        return series

    def fit(self, input_data: InputData):
        forecast_horizon = input_data.task.task_params.forecast_length
        self.model_ = MSSAForecaster(
            forecast_horizon=forecast_horizon,
            window_size=self.window_size,
            rank=self.rank,
            explained_variance=self.explained_variance,
            coupled=False,
        )
        self.model_.fit(self._prepare_series(input_data))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        series = self._prepare_series(input_data)
        prediction = self.model_.predict(series)
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData) -> np.ndarray:
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.denoised_series_, dtype=float).reshape(1, -1)
