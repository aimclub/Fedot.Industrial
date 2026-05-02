from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

try:  # pragma: no cover - benchmark/lightweight envs may not have fedot installed
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
            return type('OutputData', (),
                        {'predict': predict, 'data_type': data_type, 'idx': getattr(input_data, 'idx', None)})

    class OperationParameters(dict):  # type: ignore[override]
        def get(self, key, default=None):
            return super().get(key, default)

    class DataTypesEnum:  # pragma: no cover
        table = 'table'

from fedot_ind.core.models.ts_forecasting.dmd_models.okhs_fdmd_forecaster import (
    OKHSFDMDForecaster,
    OKHS_FDMD_DEFAULT_PARAMS,
    normalize_okhs_fdmd_params,
    normalize_okhs_fdmd_prediction,
    build_okhs_fdmd_runtime_diagnostics
)

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

# Импорт математического движка (из предыдущего шага)
from fedot_ind.core.models.kernel.deep_okhs_forecasting_torch import DeepOKHSForecasterTorch


DEEP_OKHS_FDMD_DEFAULT_PARAMS: dict[str, Any] = {
    **OKHS_FDMD_DEFAULT_PARAMS,
    'latent_dim': 16,
    'ae_epochs': 100,
    'ae_learning_rate': 1e-3,
    'alpha_adjoint': 1.0,
    'beta_rec': 1.0,
    'hidden_layers': [64, 64],
    'dt': 1.0,
}


def normalize_deep_okhs_fdmd_params(
        params: dict[str, Any] | None = None,
        *,
        forecast_horizon: int,
        series_length: int | None = None,
) -> dict[str, Any]:
    """Merge Deep OKHS defaults with user params and validate core sizes."""
    # Сначала применяем базовую нормализацию классического OKHS
    resolved = normalize_okhs_fdmd_params(
        params, 
        forecast_horizon=forecast_horizon, 
        series_length=series_length
    )
    
    # Затем добавляем гиперпараметры, специфичные для глубокого обучения
    user_params = dict(params or {})
    resolved['latent_dim'] = int(user_params.get('latent_dim', DEEP_OKHS_FDMD_DEFAULT_PARAMS['latent_dim']))
    resolved['ae_epochs'] = int(user_params.get('ae_epochs', DEEP_OKHS_FDMD_DEFAULT_PARAMS['ae_epochs']))
    resolved['ae_learning_rate'] = float(user_params.get('ae_learning_rate', DEEP_OKHS_FDMD_DEFAULT_PARAMS['ae_learning_rate']))
    resolved['alpha_adjoint'] = float(user_params.get('alpha_adjoint', DEEP_OKHS_FDMD_DEFAULT_PARAMS['alpha_adjoint']))
    resolved['beta_rec'] = float(user_params.get('beta_rec', DEEP_OKHS_FDMD_DEFAULT_PARAMS['beta_rec']))
    resolved['hidden_layers'] = list(user_params.get('hidden_layers', DEEP_OKHS_FDMD_DEFAULT_PARAMS['hidden_layers']))
    resolved['dt'] = float(user_params.get('dt', DEEP_OKHS_FDMD_DEFAULT_PARAMS['dt']))
    
    return resolved


@dataclass
class DeepOKHSFDMDForecaster(OKHSFDMDForecaster):
    """
    Stage-aware обертка для Deep OKHS fDMD. 
    Расширяет классический OKHSFDMDForecaster параметрами нейросети.
    """
    latent_dim: int = 16
    ae_epochs: int = 100
    ae_learning_rate: float = 1e-3
    alpha_adjoint: float = 1.0
    beta_rec: float = 1.0
    hidden_layers: list[int] = field(default_factory=lambda: [64, 64])
    dt: float = 1.0

    def _build_inner_model(self) -> DeepOKHSForecasterTorch:
        """Переопределяем фабричный метод для использования движка с глубоким обучением."""
        return DeepOKHSForecasterTorch(
            # Параметры классического OKHS (из базового датакласса)
            q=self.q,
            forecast_horizon=self.forecast_horizon,
            n_modes=self.n_modes,
            method='dmd',
            q_policy=self.q_policy,
            window_policy=self.window_policy,
            trajectory_sampling_policy=self.trajectory_sampling_policy,
            trajectory_rank_policy=self.trajectory_rank_policy,
            trajectory_rank_value=self.trajectory_rank_value,
            trajectory_representation_policy=self.trajectory_representation_policy,
            latent_trajectory_stride_policy=self.latent_trajectory_stride_policy,
            latent_trajectory_stride=self.latent_trajectory_stride,
            mode_selection_policy=self.mode_selection_policy,
            mode_energy_threshold=self.mode_energy_threshold,
            prediction_mode_selection_policy=self.prediction_mode_selection_policy,
            max_prediction_modes=self.max_prediction_modes,
            min_prediction_modes=self.min_prediction_modes,
            boundary_alignment_policy=self.boundary_alignment_policy,
            boundary_alignment_decay=self.boundary_alignment_decay,
            prediction_stability_threshold=self.prediction_stability_threshold,
            anti_smoothing_policy=self.anti_smoothing_policy,
            anti_smoothing_tail_window=self.anti_smoothing_tail_window,
            anti_smoothing_amplitude_ratio=self.anti_smoothing_amplitude_ratio,
            anti_smoothing_monotone_ratio=self.anti_smoothing_monotone_ratio,
            anti_smoothing_oscillation_floor=self.anti_smoothing_oscillation_floor,
            anti_smoothing_decay=self.anti_smoothing_decay,
            anti_smoothing_target_amplitude_ratio=self.anti_smoothing_target_amplitude_ratio,
            device=self.device,
            
            # Специфичные гиперпараметры Deep OKHS
            latent_dim=self.latent_dim,
            ae_epochs=self.ae_epochs,
            ae_learning_rate=self.ae_learning_rate,
            alpha_adjoint=self.alpha_adjoint,
            beta_rec=self.beta_rec,
            hidden_layers=self.hidden_layers,
            dt=self.dt
        )


@dataclass(frozen=True)
class DeepOKHSFDMDForecasterSpec:
    """Immutable construction spec for a Deep OKHS fDMD forecaster."""
    forecast_horizon: int
    params: dict[str, Any]

    @property
    def family(self) -> str:
        return 'operator_model'

    @property
    def model_name(self) -> str:
        return 'deep_okhs_fdmd_forecaster'


@dataclass(frozen=True)
class DeepOKHSFDMDForecasterRunResult:
    """Serializable run result for one Deep OKHS fDMD forecast execution."""
    spec: DeepOKHSFDMDForecasterSpec
    forecast: tuple[float, ...]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'spec': {
                'forecast_horizon': int(self.spec.forecast_horizon),
                'params': dict(self.spec.params),
                'family': self.spec.family,
                'model_name': self.spec.model_name,
            },
            'forecast': list(self.forecast),
            'diagnostics': dict(self.diagnostics),
            'metadata': dict(self.metadata),
        }


def build_deep_okhs_fdmd_spec(
        *,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
        series_length: int | None = None,
) -> DeepOKHSFDMDForecasterSpec:
    normalized_params = normalize_deep_okhs_fdmd_params(
        params,
        forecast_horizon=int(forecast_horizon),
        series_length=series_length,
    )
    normalized_params.pop('forecast_horizon', None)
    normalized_params.pop('series_length', None)
    return DeepOKHSFDMDForecasterSpec(
        forecast_horizon=int(forecast_horizon),
        params=normalized_params,
    )


def build_deep_okhs_fdmd_forecaster(
        *,
        forecast_horizon: int | None = None,
        params: dict[str, Any] | None = None,
        series_length: int | None = None,
        spec: DeepOKHSFDMDForecasterSpec | None = None,
) -> DeepOKHSFDMDForecaster:
    resolved_spec = spec or build_deep_okhs_fdmd_spec(
        forecast_horizon=int(forecast_horizon),
        params=params,
        series_length=series_length,
    )
    return DeepOKHSFDMDForecaster(
        forecast_horizon=int(resolved_spec.forecast_horizon),
        **dict(resolved_spec.params),
    )


def _operation_params_to_dict(params) -> dict[str, Any]:
    if params is None:
        return {}
    if hasattr(params, 'to_dict'):
        return dict(params.to_dict())
    return dict(params)


def run_deep_okhs_fdmd_forecaster_on_series(
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
) -> DeepOKHSFDMDForecasterRunResult:
    """Fit and forecast one series with Deep OKHS fDMD and return diagnostics."""
    history = np.asarray(time_series, dtype=float).reshape(-1)
    
    spec = build_deep_okhs_fdmd_spec(
        forecast_horizon=int(forecast_horizon),
        params=params,
        series_length=len(history),
    )
    
    model = build_deep_okhs_fdmd_forecaster(spec=spec)
    
    # Запуск Фазы 1 (Обучение) и Фазы 2 (DMD)
    model.fit(history)
    
    # Инференс
    forecast = normalize_okhs_fdmd_prediction(
        model.predict(history),
        forecast_horizon=spec.forecast_horizon,
    )
    
    return DeepOKHSFDMDForecasterRunResult(
        spec=spec,
        forecast=tuple(float(value) for value in forecast.tolist()),
        diagnostics=model.get_diagnostics(),
        metadata={
            'history_length': int(len(history)),
            'model_family': spec.family,
            'resolved_params': dict(spec.params),
        },
    )


class DeepOKHSFDMDForecasterImplementation(ModelImplementation):
    """FEDOT-compatible implementation wrapper for DeepOKHSFDMDForecaster."""
    
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.model_: DeepOKHSFDMDForecaster | None = None

    def fit(self, input_data: InputData):
        features = np.asarray(input_data.features, dtype=float)
        params = _operation_params_to_dict(self.params)
        
        self.spec_ = build_deep_okhs_fdmd_spec(
            forecast_horizon=input_data.task.task_params.forecast_length,
            params=params,
            series_length=int(np.asarray(features).reshape(-1).shape[0]),
        )
        self.model_ = build_deep_okhs_fdmd_forecaster(spec=self.spec_)
        self.model_.fit(features)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        prediction = normalize_okhs_fdmd_prediction(
            self.model_.predict(np.asarray(input_data.features, dtype=float)),
            forecast_horizon=int(self.model_.forecast_horizon),
        )
        return self._convert_to_output(
            input_data,
            predict=prediction,
            data_type=DataTypesEnum.table,
        )

    def predict_for_fit(self, input_data: InputData):
        if self.model_ is None:
            self.fit(input_data)
        return self.predict(input_data)

    def get_diagnostics(self) -> dict[str, Any]:
        if self.model_ is None:
            return {}
        return self.model_.get_diagnostics()

    def get_stage_tuning_plan(self) -> dict[str, object]:
        return build_forecasting_stage_tuning_plan(
            'deep_okhs_fdmd_forecaster',
            _operation_params_to_dict(self.params),
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'deep_okhs_fdmd_forecaster',
                _operation_params_to_dict(self.params),
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return build_forecasting_stage_tuning_execution(
            'deep_okhs_fdmd_forecaster',
            base_params=_operation_params_to_dict(self.params),
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning(self, objective, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return run_sequential_stage_tuning(
            'deep_okhs_fdmd_forecaster',
            objective=objective,
            base_params=_operation_params_to_dict(self.params),
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
            'deep_okhs_fdmd_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params=_operation_params_to_dict(self.params),
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()