from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - benchmark/lightweight envs may not have fedot installed
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

        def _convert_to_output(self, input_data, predict=None, data_type=None):
            return type('OutputData', (),
                        {'predict': predict, 'data_type': data_type, 'idx': getattr(input_data, 'idx', None)})

    class OperationParameters(dict):  # type: ignore[override]
        def get(self, key, default=None):
            return super().get(key, default)

    class DataTypesEnum:  # pragma: no cover - only used in full FEDOT runtime
        table = 'table'

from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
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


from industrial_forecasting.neural.hybrid import HybridEnsembleForecaster as _HybridEnsembleForecaster


class HybridEnsembleForecaster(_HybridEnsembleForecaster):
    """Supply the Industrial OKHS branch without adding it to the standalone wheel."""

    def _build_complex_branch(self):
        branch = str(self.complex_branch).lower()
        if branch == 'havok':
            return super()._build_complex_branch()
        if branch == 'okhs':
            return OKHSForecaster(forecast_horizon=self.forecast_horizon, **self.complex_params)
        raise ValueError("Industrial hybrid forecasting supports complex_branch='havok' or 'okhs'.")


class HybridEnsembleForecasterImplementation(ModelImplementation):
    """FEDOT-compatible wrapper for HybridEnsembleForecaster."""

    def __init__(self, params: Optional[OperationParameters] = None):
        """Read ensemble branch and calibration parameters from operation params."""
        params = params or OperationParameters()
        super().__init__(params)
        self.complex_branch = str(self.params.get('complex_branch', 'havok'))
        self.branch_failure_policy = str(self.params.get('branch_failure_policy', 'raise'))
        self.calibration_horizon = self.params.get('calibration_horizon')
        self.device = str(self.params.get('device', 'auto'))
        self.lagged_params = dict(self.params.get('lagged_params', {}))
        self.low_rank_params = dict(self.params.get('low_rank_params', {}))
        self.complex_params = dict(self.params.get('complex_params', {}))
        self.model_: HybridEnsembleForecaster | None = None

    def fit(self, input_data: InputData):
        """Fit the wrapped hybrid ensemble from FEDOT InputData."""
        self.model_ = HybridEnsembleForecaster(
            forecast_horizon=input_data.task.task_params.forecast_length,
            complex_branch=self.complex_branch,
            branch_failure_policy=self.branch_failure_policy,
            calibration_horizon=self.calibration_horizon,
            device=self.device,
            lagged_params=self.lagged_params,
            low_rank_params=self.low_rank_params,
            complex_params=self.complex_params,
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Return FEDOT OutputData with the ensemble forecast."""
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        """Return ensemble predictions for fit-time compatibility paths."""
        if self.model_ is None:
            self.fit(input_data)
        return np.asarray(self.model_.predict(np.asarray(input_data.features, dtype=float)), dtype=float)

    def get_diagnostics(self) -> dict[str, object]:
        """Expose diagnostics from the fitted wrapped model."""
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        """Return the stage-aware tuning plan for the hybrid ensemble."""
        return build_forecasting_stage_tuning_plan(
            'hybrid_ensemble_forecaster',
            {
                'complex_branch': self.complex_branch,
                'branch_failure_policy': self.branch_failure_policy,
                'calibration_horizon': self.calibration_horizon,
                'lagged_params': self.lagged_params,
                'low_rank_params': self.low_rank_params,
                'complex_params': self.complex_params,
            },
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        """Return hybrid ensemble search-space slices grouped by stage."""
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'hybrid_ensemble_forecaster',
                {
                    'complex_branch': self.complex_branch,
                    'branch_failure_policy': self.branch_failure_policy,
                    'calibration_horizon': self.calibration_horizon,
                    'lagged_params': self.lagged_params,
                    'low_rank_params': self.low_rank_params,
                    'complex_params': self.complex_params,
                },
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        """Resolve proposed ensemble updates into a stage tuning execution."""
        return build_forecasting_stage_tuning_execution(
            'hybrid_ensemble_forecaster',
            base_params={
                'complex_branch': self.complex_branch,
                'branch_failure_policy': self.branch_failure_policy,
                'calibration_horizon': self.calibration_horizon,
                'lagged_params': self.lagged_params,
                'low_rank_params': self.low_rank_params,
                'complex_params': self.complex_params,
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
        """Run runtime stage tuning for the ensemble on a raw time series."""
        return run_forecasting_stage_tuning_on_series(
            'hybrid_ensemble_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params={
                'complex_branch': self.complex_branch,
                'branch_failure_policy': self.branch_failure_policy,
                'calibration_horizon': self.calibration_horizon,
                'device': self.device,
                'lagged_params': self.lagged_params,
                'low_rank_params': self.low_rank_params,
                'complex_params': self.complex_params,
            },
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
