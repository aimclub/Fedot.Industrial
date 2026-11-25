from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - benchmark and lightweight environments may not have fedot installed
    from fedot.core.data.input_data.data import InputData, OutputData
    from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
        ModelImplementation,
    )
    from fedot.core.operations.operation_parameters import OperationParameters
    from fedot.core.repository.dataset_types import DataTypesEnum
except Exception:  # pragma: no cover
    InputData = OutputData = None

    class ModelImplementation:  # type: ignore[override]
        def __init__(self, params=None):
            self.params = params or {}

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


from industrial_forecasting.neural.mssa import (
    MSSAForecaster as MSSAForecaster,
)


class MSSAForecasterImplementation(ModelImplementation):
    """FEDOT-compatible wrapper for MSSAForecaster."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Read mSSA stage and head parameters from operation params."""
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = self.params.get('window_size')
        self.rank = self.params.get('rank')
        self.explained_variance = self.params.get('explained_variance', 0.95)
        self.lag_order = self.params.get('lag_order')
        self.coupled = not self.params.get('channel_independent', False)
        self.head_policy = str(self.params.get('head_policy', 'mlp'))
        self.head_hidden_dim = int(self.params.get('head_hidden_dim', 64))
        self.head_hidden_layers = int(self.params.get('head_hidden_layers', 2))
        self.head_epochs = int(self.params.get('head_epochs', 120))
        self.head_learning_rate = float(self.params.get('head_learning_rate', 1e-3))
        self.head_random_seed = int(self.params.get('head_random_seed', 42))
        self.device = str(self.params.get('device', 'auto'))
        self.progress_policy = self.params.get('progress_policy')
        self.model_: MSSAForecaster | None = None

    def fit(self, input_data: InputData):
        """Fit the wrapped mSSA model from FEDOT InputData."""
        forecast_horizon = input_data.task.task_params.forecast_length
        self.model_ = MSSAForecaster(
            forecast_horizon=forecast_horizon,
            window_size=self.window_size,
            rank=self.rank,
            explained_variance=self.explained_variance,
            lag_order=self.lag_order,
            coupled=self.coupled,
            head_policy=self.head_policy,
            head_hidden_dim=self.head_hidden_dim,
            head_hidden_layers=self.head_hidden_layers,
            head_epochs=self.head_epochs,
            head_learning_rate=self.head_learning_rate,
            head_random_seed=self.head_random_seed,
            device=self.device,
            progress_policy=self.progress_policy,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the mSSA forecast."""
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData) -> np.ndarray:
        """Return denoised series features for fit-time compatibility paths."""
        if self.model_ is None:
            forecast_horizon = input_data.task.task_params.forecast_length
            self.model_ = MSSAForecaster(
                forecast_horizon=forecast_horizon,
                window_size=self.window_size,
                rank=self.rank,
                explained_variance=self.explained_variance,
                lag_order=self.lag_order,
                coupled=self.coupled,
                head_policy=self.head_policy,
                head_hidden_dim=self.head_hidden_dim,
                head_hidden_layers=self.head_hidden_layers,
                head_epochs=self.head_epochs,
                head_learning_rate=self.head_learning_rate,
                head_random_seed=self.head_random_seed,
                device=self.device,
                progress_policy=self.progress_policy,
            ).fit(np.asarray(input_data.features, dtype=float))
        denoised = np.asarray(self.model_.denoised_series_, dtype=float)
        return denoised.T if denoised.ndim > 1 else denoised.reshape(1, -1)

    def get_diagnostics(self) -> dict[str, object]:
        """Expose diagnostics from the fitted wrapped model."""
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        """Return the stage-aware tuning plan for mSSA."""
        return build_forecasting_stage_tuning_plan(
            'mssa_forecaster',
            {
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'lag_order': self.lag_order,
                'channel_independent': not self.coupled,
                'head_policy': self.head_policy,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'head_random_seed': self.head_random_seed,
                'device': self.device,
            },
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        """Return search-space slices grouped by mSSA stages."""
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'mssa_forecaster',
                {
                    'window_size': self.window_size,
                    'rank': self.rank,
                    'explained_variance': self.explained_variance,
                    'lag_order': self.lag_order,
                    'channel_independent': not self.coupled,
                    'head_policy': self.head_policy,
                    'head_hidden_dim': self.head_hidden_dim,
                    'head_hidden_layers': self.head_hidden_layers,
                    'head_epochs': self.head_epochs,
                    'head_learning_rate': self.head_learning_rate,
                    'head_random_seed': self.head_random_seed,
                    'device': self.device,
                },
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve proposed mSSA updates into a stage tuning execution."""
        return build_forecasting_stage_tuning_execution(
            'mssa_forecaster',
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'lag_order': self.lag_order,
                'channel_independent': not self.coupled,
                'head_policy': self.head_policy,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'head_random_seed': self.head_random_seed,
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
        """Run runtime stage tuning for mSSA on a raw time series."""
        return run_forecasting_stage_tuning_on_series(
            'mssa_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'window_size': self.window_size,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'lag_order': self.lag_order,
                'channel_independent': not self.coupled,
                'head_policy': self.head_policy,
                'head_hidden_dim': self.head_hidden_dim,
                'head_hidden_layers': self.head_hidden_layers,
                'head_epochs': self.head_epochs,
                'head_learning_rate': self.head_learning_rate,
                'head_random_seed': self.head_random_seed,
                'device': self.device,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
