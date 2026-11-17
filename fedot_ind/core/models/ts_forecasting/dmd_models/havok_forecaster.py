from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - benchmark/lightweight envs may not have fedot installed
    from fedot.core.data.input_data.data import InputData, OutputData
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

from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime import (
    run_forecasting_stage_tuning_on_series,
)


from industrial_forecasting.neural.havok import (
    HAVOKForecaster as HAVOKForecaster,
)


class HAVOKForecasterImplementation(ModelImplementation):
    """FEDOT-compatible wrapper for HAVOKForecaster."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Read HAVOK stage and head parameters from operation params."""
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = self.params.get('window_size')
        self.rank = self.params.get('rank')
        self.forcing_threshold_scale = self.params.get('forcing_threshold_scale', 1.0)
        self.forcing_decay = self.params.get('forcing_decay', 0.85)
        self.head_policy = str(self.params.get('head_policy', 'mlp'))
        self.head_activation = str(self.params.get('head_activation', 'relu'))
        self.head_depth = int(self.params.get('head_depth', 2))
        raw_head_base_hidden_dim = self.params.get('head_base_hidden_dim')
        self.head_base_hidden_dim = 512 if raw_head_base_hidden_dim is None else int(raw_head_base_hidden_dim)
        raw_head_hidden_dim = self.params.get('head_hidden_dim')
        self.head_hidden_dim = None if raw_head_hidden_dim is None else int(raw_head_hidden_dim)
        raw_head_hidden_layers = self.params.get('head_hidden_layers')
        self.head_hidden_layers = None if raw_head_hidden_layers is None else int(raw_head_hidden_layers)
        self.head_epochs = int(self.params.get('head_epochs', 120))
        self.head_learning_rate = float(self.params.get('head_learning_rate', 1e-3))
        self.device = str(self.params.get('device', 'auto'))
        self.progress_policy = self.params.get('progress_policy')
        self.model_: HAVOKForecaster | None = None

    def fit(self, input_data: InputData):
        """Fit the wrapped HAVOK forecaster from FEDOT InputData."""
        self.model_ = HAVOKForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=self.window_size,
            rank=self.rank,
            forcing_threshold_scale=self.forcing_threshold_scale,
            forcing_decay=self.forcing_decay,
            head_policy=self.head_policy,
            head_activation=self.head_activation,
            head_depth=self.head_depth,
            head_base_hidden_dim=self.head_base_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            head_hidden_layers=self.head_hidden_layers,
            head_epochs=self.head_epochs,
            head_learning_rate=self.head_learning_rate,
            device=self.device,
            progress_policy=self.progress_policy,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the HAVOK forecast."""
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        """Return latent states for fit-time compatibility paths."""
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.latent_states_, dtype=float).T

    def get_diagnostics(self) -> dict[str, object]:
        """Expose diagnostics from the fitted wrapped model."""
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        """Return the stage-aware tuning plan for HAVOK."""
        return build_forecasting_stage_tuning_plan(
            'havok_forecaster',
            {
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
            },
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        """Return HAVOK search-space slices grouped by stage."""
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'havok_forecaster',
                {
                    'window_size': self.window_size,
                    'rank': self.rank,
                    'forcing_threshold_scale': self.forcing_threshold_scale,
                    'forcing_decay': self.forcing_decay,
                    'head_policy': self.head_policy,
                    'head_activation': self.head_activation,
                    'head_depth': self.head_depth,
                    'head_base_hidden_dim': self.head_base_hidden_dim,
                    'head_hidden_dim': self.head_hidden_dim,
                    'head_hidden_layers': self.head_hidden_layers,
                    'head_epochs': self.head_epochs,
                    'head_learning_rate': self.head_learning_rate,
                    'device': self.device,
                },
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve proposed HAVOK updates into a stage tuning execution."""
        return build_forecasting_stage_tuning_execution(
            'havok_forecaster',
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
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
        """Run runtime stage tuning for HAVOK on a raw time series."""
        return run_forecasting_stage_tuning_on_series(
            'havok_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'forcing_threshold_scale': self.forcing_threshold_scale,
                'forcing_decay': self.forcing_decay,
                'head_policy': self.head_policy,
                'head_activation': self.head_activation,
                'head_depth': self.head_depth,
                'head_base_hidden_dim': self.head_base_hidden_dim,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'device': self.device,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
