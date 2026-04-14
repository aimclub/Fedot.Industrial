from __future__ import annotations

from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.models.ts_forecasting.mssa_forecaster import MSSAForecaster
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import build_forecasting_stage_tuning_execution
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import run_forecasting_stage_tuning_on_series


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

    def get_diagnostics(self) -> dict[str, object]:
        diagnostics = {
            'compatibility_status': self.compatibility_status_,
            'history_lookback': int(self.history_lookback),
            'mode': self.mode,
        }
        if self.model_ is not None and hasattr(self.model_, 'get_diagnostics'):
            diagnostics.update(self.model_.get_diagnostics())
        return diagnostics

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'ssa_forecaster',
            {
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'history_lookback': self.history_lookback,
                'mode': self.mode,
            },
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'ssa_forecaster',
                {
                    'window_size': self.window_size,
                    'rank': self.rank,
                    'explained_variance': self.explained_variance,
                    'history_lookback': self.history_lookback,
                    'mode': self.mode,
                },
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return build_forecasting_stage_tuning_execution(
            'ssa_forecaster',
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'history_lookback': self.history_lookback,
                'mode': self.mode,
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
            'ssa_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'history_lookback': self.history_lookback,
                'mode': self.mode,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
