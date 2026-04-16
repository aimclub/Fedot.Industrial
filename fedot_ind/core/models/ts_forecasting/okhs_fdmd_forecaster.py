from __future__ import annotations

from dataclasses import dataclass
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


    class DataTypesEnum:  # pragma: no cover - only used in full FEDOT runtime
        table = 'table'

from fedot_ind.core.models.kernel.okhs_forecasting import OKHSForecaster
from fedot_ind.core.models.kernel.okhs_runtime import build_okhs_stage_diagnostics
from fedot_ind.core.models.ts_forecasting.stage_tuning import (
    build_forecasting_stage_search_spaces,
    build_forecasting_stage_tuning_plan,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_execution import (
    build_forecasting_stage_tuning_execution,
    run_sequential_stage_tuning,
)
from fedot_ind.core.models.ts_forecasting.stage_tuning_runtime import run_forecasting_stage_tuning_on_series

OKHS_FDMD_DEFAULT_PARAMS: dict[str, Any] = {
    'q': 0.7,
    'n_modes': 5,
    'window_size': 20,
    'q_policy': 'fixed',
    'window_policy': 'adaptive_cycle_aware',
    'trajectory_sampling_policy': 'dense',
    'trajectory_rank_policy': 'explained_dispersion',
    'trajectory_rank_value': None,
    'trajectory_representation_policy': 'projected',
    'latent_trajectory_stride_policy': 'adaptive',
    'latent_trajectory_stride': None,
    'mode_selection_policy': 'energy',
    'mode_energy_threshold': 0.95,
    'prediction_mode_selection_policy': 'adaptive_tail_energy',
    'max_prediction_modes': None,
    'min_prediction_modes': 4,
    'boundary_alignment_policy': 'tapered_offset',
    'boundary_alignment_decay': 4.0,
    'prediction_stability_threshold': 0.03,
    'anti_smoothing_policy': 'residual_bridge',
    'anti_smoothing_tail_window': None,
    'anti_smoothing_amplitude_ratio': 0.35,
    'anti_smoothing_monotone_ratio': 0.9,
    'anti_smoothing_oscillation_floor': 0.25,
    'anti_smoothing_decay': 2.5,
    'anti_smoothing_target_amplitude_ratio': 0.8,
    'device': 'cpu',
}


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def normalize_okhs_fdmd_params(
        params: dict[str, Any] | None = None,
        *,
        forecast_horizon: int,
        series_length: int | None = None,
) -> dict[str, Any]:
    resolved = {
        key: value
        for key, value in OKHS_FDMD_DEFAULT_PARAMS.items()
    }
    for key, value in dict(params or {}).items():
        if value is not None:
            resolved[key] = value

    resolved_forecast_horizon = int(forecast_horizon)
    resolved_window_size = int(resolved.get('window_size', 20))
    if series_length is None:
        resolved['window_size'] = max(4, resolved_window_size)
    else:
        max_window_size = max(2, int(series_length) - 1)
        min_window_size = 4 if max_window_size >= 4 else max_window_size
        resolved['window_size'] = min(max(resolved_window_size, min_window_size), max_window_size)

    resolved['q'] = float(resolved.get('q', 0.7))
    resolved['n_modes'] = int(resolved.get('n_modes', 5))
    resolved['q_policy'] = str(resolved.get('q_policy', 'fixed'))
    resolved['window_policy'] = str(resolved.get('window_policy', 'adaptive_cycle_aware'))
    resolved['trajectory_sampling_policy'] = str(resolved.get('trajectory_sampling_policy', 'dense'))
    resolved['trajectory_rank_policy'] = str(resolved.get('trajectory_rank_policy', 'explained_dispersion'))
    resolved['trajectory_rank_value'] = _maybe_int(resolved.get('trajectory_rank_value'))
    resolved['trajectory_representation_policy'] = str(
        resolved.get('trajectory_representation_policy', 'projected')
    )
    resolved['latent_trajectory_stride_policy'] = str(
        resolved.get('latent_trajectory_stride_policy', 'adaptive')
    )
    resolved['latent_trajectory_stride'] = _maybe_int(resolved.get('latent_trajectory_stride'))
    resolved['mode_selection_policy'] = str(resolved.get('mode_selection_policy', 'energy'))
    resolved['mode_energy_threshold'] = float(resolved.get('mode_energy_threshold', 0.95))
    resolved['prediction_mode_selection_policy'] = str(
        resolved.get('prediction_mode_selection_policy', 'adaptive_tail_energy')
    )
    resolved['max_prediction_modes'] = _maybe_int(resolved.get('max_prediction_modes'))
    resolved['min_prediction_modes'] = int(resolved.get('min_prediction_modes', 4))
    resolved['boundary_alignment_policy'] = str(resolved.get('boundary_alignment_policy', 'tapered_offset'))
    resolved['boundary_alignment_decay'] = float(resolved.get('boundary_alignment_decay', 4.0))
    resolved['prediction_stability_threshold'] = _maybe_float(resolved.get('prediction_stability_threshold'))
    resolved['anti_smoothing_policy'] = str(resolved.get('anti_smoothing_policy', 'residual_bridge'))
    resolved['anti_smoothing_tail_window'] = _maybe_int(resolved.get('anti_smoothing_tail_window'))
    resolved['anti_smoothing_amplitude_ratio'] = float(resolved.get('anti_smoothing_amplitude_ratio', 0.35))
    resolved['anti_smoothing_monotone_ratio'] = float(resolved.get('anti_smoothing_monotone_ratio', 0.9))
    resolved['anti_smoothing_oscillation_floor'] = float(
        resolved.get('anti_smoothing_oscillation_floor', 0.25)
    )
    resolved['anti_smoothing_decay'] = float(resolved.get('anti_smoothing_decay', 2.5))
    resolved['anti_smoothing_target_amplitude_ratio'] = float(
        resolved.get('anti_smoothing_target_amplitude_ratio', 0.8)
    )
    resolved['device'] = str(resolved.get('device', 'cpu'))
    resolved['forecast_horizon'] = resolved_forecast_horizon
    if series_length is not None:
        resolved['series_length'] = int(series_length)
    return resolved


def normalize_okhs_fdmd_prediction(
        prediction: np.ndarray | Any,
        *,
        forecast_horizon: int,
) -> np.ndarray:
    if hasattr(prediction, 'cpu'):
        prediction = prediction.cpu()
    return np.asarray(prediction, dtype=float).reshape(-1)[: int(forecast_horizon)]


def build_okhs_fdmd_runtime_diagnostics(
        optimization_info: dict[str, Any],
        *,
        model_name: str = 'okhs_fdmd_forecaster',
        model_family: str = 'operator_model',
        prediction_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_diagnostics = build_okhs_stage_diagnostics(dict(optimization_info))
    runtime_diagnostics = {
        'model_family': model_family,
        'model_name': model_name,
        **stage_diagnostics,
        'raw_okhs': dict(optimization_info),
    }
    if prediction_diagnostics:
        runtime_diagnostics.update(dict(prediction_diagnostics))
    return runtime_diagnostics


@dataclass
class OKHSFDMDForecaster:
    forecast_horizon: int
    q: float = 0.7
    n_modes: int = 5
    window_size: int = 20
    q_policy: str = 'fixed'
    window_policy: str = 'adaptive_cycle_aware'
    trajectory_sampling_policy: str = 'dense'
    trajectory_rank_policy: str = 'explained_dispersion'
    trajectory_rank_value: int | None = None
    trajectory_representation_policy: str = 'projected'
    latent_trajectory_stride_policy: str = 'adaptive'
    latent_trajectory_stride: int | None = None
    mode_selection_policy: str = 'energy'
    mode_energy_threshold: float = 0.95
    prediction_mode_selection_policy: str = 'adaptive_tail_energy'
    max_prediction_modes: int | None = None
    min_prediction_modes: int = 4
    boundary_alignment_policy: str = 'tapered_offset'
    boundary_alignment_decay: float = 4.0
    prediction_stability_threshold: float | None = 0.03
    anti_smoothing_policy: str = 'residual_bridge'
    anti_smoothing_tail_window: int | None = None
    anti_smoothing_amplitude_ratio: float = 0.35
    anti_smoothing_monotone_ratio: float = 0.9
    anti_smoothing_oscillation_floor: float = 0.25
    anti_smoothing_decay: float = 2.5
    anti_smoothing_target_amplitude_ratio: float = 0.8
    device: str = 'cpu'

    def _build_inner_model(self) -> OKHSForecaster:
        return OKHSForecaster(
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
        )

    def _refresh_diagnostics(self):
        optimization_info = dict(self.inner_model_.get_optimization_info())
        self.raw_diagnostics_ = optimization_info
        self.stage_diagnostics_ = build_okhs_stage_diagnostics(optimization_info)
        self.diagnostics_ = build_okhs_fdmd_runtime_diagnostics(optimization_info)

    def fit(self, time_series: np.ndarray) -> 'OKHSFDMDForecaster':
        series = np.asarray(time_series, dtype=float).reshape(-1)
        self.inner_model_ = self._build_inner_model()
        self.inner_model_.fit(series, window_size=self.window_size)
        self.training_history_ = series
        self._refresh_diagnostics()
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon > self.forecast_horizon:
            raise ValueError(
                f'OKHSFDMDForecaster was trained for horizon={self.forecast_horizon}, got requested horizon={horizon}.'
            )
        source_series = self.training_history_ if time_series is None else np.asarray(time_series, dtype=float).reshape(
            -1)
        self.last_prediction_diagnostics_ = {
            'source_series_length': int(len(source_series)),
        }
        forecast = self.inner_model_.predict(source_series)
        values = normalize_okhs_fdmd_prediction(forecast, forecast_horizon=horizon)
        self._refresh_diagnostics()
        self.last_prediction_diagnostics_.update({
            'forecast_shape': tuple(int(value) for value in values.shape),
            'first_prediction_value': float(values[0]) if len(values) else None,
        })
        return values

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            **self.diagnostics_,
            **getattr(self, 'last_prediction_diagnostics_', {}),
        }


@dataclass(frozen=True)
class OKHSFDMDForecasterSpec:
    forecast_horizon: int
    params: dict[str, Any]

    @property
    def family(self) -> str:
        return 'operator_model'

    @property
    def model_name(self) -> str:
        return 'okhs_fdmd_forecaster'


@dataclass(frozen=True)
class OKHSFDMDForecasterRunResult:
    spec: OKHSFDMDForecasterSpec
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


def build_okhs_fdmd_spec(
        *,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
        series_length: int | None = None,
) -> OKHSFDMDForecasterSpec:
    normalized_params = normalize_okhs_fdmd_params(
        params,
        forecast_horizon=int(forecast_horizon),
        series_length=series_length,
    )
    normalized_params.pop('forecast_horizon', None)
    normalized_params.pop('series_length', None)
    return OKHSFDMDForecasterSpec(
        forecast_horizon=int(forecast_horizon),
        params=normalized_params,
    )


def build_okhs_fdmd_forecaster(
        *,
        forecast_horizon: int | None = None,
        params: dict[str, Any] | None = None,
        series_length: int | None = None,
        spec: OKHSFDMDForecasterSpec | None = None,
) -> OKHSFDMDForecaster:
    resolved_spec = spec or build_okhs_fdmd_spec(
        forecast_horizon=int(forecast_horizon),
        params=params,
        series_length=series_length,
    )
    return OKHSFDMDForecaster(
        forecast_horizon=int(resolved_spec.forecast_horizon),
        **dict(resolved_spec.params),
    )


def run_okhs_fdmd_forecaster_on_series(
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
) -> OKHSFDMDForecasterRunResult:
    history = np.asarray(time_series, dtype=float).reshape(-1)
    spec = build_okhs_fdmd_spec(
        forecast_horizon=int(forecast_horizon),
        params=params,
        series_length=len(history),
    )
    model = build_okhs_fdmd_forecaster(
        spec=spec,
    )
    model.fit(history)
    forecast = normalize_okhs_fdmd_prediction(
        model.predict(history),
        forecast_horizon=spec.forecast_horizon,
    )
    return OKHSFDMDForecasterRunResult(
        spec=spec,
        forecast=tuple(float(value) for value in forecast.tolist()),
        diagnostics=model.get_diagnostics(),
        metadata={
            'history_length': int(len(history)),
            'model_family': spec.family,
            'resolved_params': dict(spec.params),
        },
    )


class OKHSFDMDForecasterImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.model_: OKHSFDMDForecaster | None = None

    def fit(self, input_data: InputData):
        features = np.asarray(input_data.features, dtype=float)
        self.spec_ = build_okhs_fdmd_spec(
            forecast_horizon=input_data.task.task_params.forecast_length,
            params=dict(self.params),
            series_length=int(np.asarray(features).reshape(-1).shape[0]),
        )
        self.model_ = build_okhs_fdmd_forecaster(spec=self.spec_)
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
            'okhs_fdmd_forecaster',
            dict(self.params),
        ).to_dict()

    def get_stage_search_spaces(self) -> tuple[dict[str, object], ...]:
        return tuple(
            stage.to_dict() for stage in build_forecasting_stage_search_spaces(
                'okhs_fdmd_forecaster',
                dict(self.params),
            )
        )

    def get_stage_tuning_execution(self, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return build_forecasting_stage_tuning_execution(
            'okhs_fdmd_forecaster',
            base_params=dict(self.params),
            stage_updates=stage_updates,
        ).to_dict()

    def run_stage_tuning(self, objective, stage_updates: dict[str, object] | None = None) -> dict[str, object]:
        return run_sequential_stage_tuning(
            'okhs_fdmd_forecaster',
            objective=objective,
            base_params=dict(self.params),
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
            'okhs_fdmd_forecaster',
            time_series=np.asarray(time_series, dtype=float),
            forecast_horizon=int(forecast_horizon),
            base_params=dict(self.params),
            stage_updates=stage_updates,
            metric_name=metric_name,
            split_spec=split_spec,
            seasonal_period=seasonal_period,
            max_values_per_parameter=max_values_per_parameter,
            max_stage_candidates=max_stage_candidates,
        ).to_dict()
