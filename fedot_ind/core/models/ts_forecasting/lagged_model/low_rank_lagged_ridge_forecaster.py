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
    run_sequential_stage_tuning,
)
from fedot_ind.core.models.ts_forecasting.forecast_tuning.stage_tuning_runtime import (
    run_forecasting_stage_tuning_on_series,
)


from industrial_forecasting.neural.low_rank import (
    LowRankLaggedRidgeForecaster as LowRankLaggedRidgeForecaster,
)


class LowRankLaggedRidgeForecasterImplementation(ModelImplementation):
    """FEDOT-compatible wrapper for LowRankLaggedRidgeForecaster."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Read low-rank lagged ridge parameters from FEDOT operation params."""
        params = params or OperationParameters()
        super().__init__(params)
        self.window_size = self.params.get('window_size')
        self.has_explicit_window_percent_ = 'window_size_percent' in self.params.keys()
        self.window_size_percent = self.params.get('window_size_percent')
        self.stride = int(self.params.get('stride', 1))
        self.alpha = float(self.params.get('alpha', 1.0))
        self.rank = self.params.get('rank')
        self.explained_variance = float(self.params.get('explained_variance', 0.95))
        self.decomposition_strategy = str(self.params.get('decomposition_strategy', 'full'))
        self.rank_truncation_policy = str(self.params.get('rank_truncation_policy', 'explained_variance'))
        self.unfolding_strategy = str(self.params.get('unfolding_strategy', 'channels_last'))
        self.device = str(self.params.get('device', 'auto'))
        self.model_: LowRankLaggedRidgeForecaster | None = None

    def fit(self, input_data: InputData):
        """Fit the wrapped low-rank lagged ridge model from InputData."""
        self.model_ = LowRankLaggedRidgeForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            window_size=None if self.has_explicit_window_percent_ else self.window_size,
            window_size_percent=self.window_size_percent if self.has_explicit_window_percent_ else None,
            stride=self.stride,
            alpha=self.alpha,
            rank=self.rank,
            explained_variance=self.explained_variance,
            decomposition_strategy=self.decomposition_strategy,
            rank_truncation_policy=self.rank_truncation_policy,
            unfolding_strategy=self.unfolding_strategy,
            device=self.device,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the low-rank lagged forecast."""
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        """Return projected low-rank features for fit-time compatibility paths."""
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.rank_result_.projected_features.detach().cpu().numpy(), dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        """Expose diagnostics from the fitted wrapped model."""
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        """Return the stage-aware tuning plan for this operation."""
        base_params = {
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'stride': self.stride,
            'alpha': self.alpha,
            'rank': self.rank,
            'explained_variance': self.explained_variance,
            'decomposition_strategy': self.decomposition_strategy,
            'rank_truncation_policy': self.rank_truncation_policy,
            'unfolding_strategy': self.unfolding_strategy,
            'device': self.device,
        }
        return build_forecasting_stage_tuning_plan(
            'low_rank_lagged_ridge_forecaster',
            base_params,
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        """Return search-space slices grouped by forecasting stage."""
        base_params = {
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'stride': self.stride,
            'alpha': self.alpha,
            'rank': self.rank,
            'explained_variance': self.explained_variance,
            'decomposition_strategy': self.decomposition_strategy,
            'rank_truncation_policy': self.rank_truncation_policy,
            'unfolding_strategy': self.unfolding_strategy,
            'device': self.device,
        }
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'low_rank_lagged_ridge_forecaster',
                base_params,
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve proposed updates into a concrete stage tuning execution."""
        base_params = {
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'stride': self.stride,
            'alpha': self.alpha,
            'rank': self.rank,
            'explained_variance': self.explained_variance,
            'decomposition_strategy': self.decomposition_strategy,
            'rank_truncation_policy': self.rank_truncation_policy,
            'unfolding_strategy': self.unfolding_strategy,
            'device': self.device,
        }
        return build_forecasting_stage_tuning_execution(
            'low_rank_lagged_ridge_forecaster',
            base_params=base_params,
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning(self, objective, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Run sequential stage tuning with an externally supplied objective."""
        base_params = {
            'window_size': self.window_size,
            'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
            'stride': self.stride,
            'alpha': self.alpha,
            'rank': self.rank,
            'explained_variance': self.explained_variance,
            'decomposition_strategy': self.decomposition_strategy,
            'rank_truncation_policy': self.rank_truncation_policy,
            'unfolding_strategy': self.unfolding_strategy,
            'device': self.device,
        }
        return run_sequential_stage_tuning(
            'low_rank_lagged_ridge_forecaster',
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
        """Run runtime stage tuning directly on a raw time series."""
        return run_forecasting_stage_tuning_on_series(
            'low_rank_lagged_ridge_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'window_size': self.window_size,
                'window_size_percent': self.window_size_percent if self.has_explicit_window_percent_ else None,
                'stride': self.stride,
                'alpha': self.alpha,
                'rank': self.rank,
                'explained_variance': self.explained_variance,
                'decomposition_strategy': self.decomposition_strategy,
                'rank_truncation_policy': self.rank_truncation_policy,
                'unfolding_strategy': self.unfolding_strategy,
                'device': self.device,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
