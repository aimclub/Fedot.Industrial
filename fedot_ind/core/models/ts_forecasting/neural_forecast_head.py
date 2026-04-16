from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name

NEURAL_FORECASTING_MODEL_REGISTRY: dict[str, Any] = {
    'patch_tst_model': (
        'fedot_ind.core.models.nn.network_impl.forecasting_model.patch_tst',
        'PatchTSTModel',
    ),
    'tcn_model': (
        'fedot_ind.core.models.nn.network_impl.forecasting_model.deep_tcn',
        'TCNModel',
    ),
    'deepar_model': (
        'fedot_ind.core.models.nn.network_impl.forecasting_model.deepar',
        'DeepAR',
    ),
    'nbeats_model': (
        'fedot_ind.core.models.nn.network_impl.forecasting_model.nbeats',
        'NBeatsModel',
    ),
}


@dataclass(frozen=True)
class NeuralForecastHeadSpec:
    model_name: str
    forecast_horizon: int
    params: dict[str, Any] | None = None

    def __post_init__(self):
        canonical_name = canonical_forecasting_model_name(self.model_name)
        if canonical_name not in NEURAL_FORECASTING_MODEL_REGISTRY:
            raise ValueError(f'Unsupported neural forecasting head: {self.model_name}')
        object.__setattr__(self, 'model_name', canonical_name)
        object.__setattr__(self, 'forecast_horizon', int(self.forecast_horizon))
        object.__setattr__(self, 'params', dict(self.params or {}))

    @property
    def family(self) -> str:
        return 'neural_forecaster'


def resolve_neural_forecasting_model_cls(model_name: str):
    entry = NEURAL_FORECASTING_MODEL_REGISTRY[model_name]
    if isinstance(entry, tuple):
        module = importlib.import_module(entry[0])
        return getattr(module, entry[1])
    return entry


def build_neural_forecasting_stage_diagnostics(
        model_name: str,
        *,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
        training_history_length: int | None = None,
) -> dict[str, Any]:
    canonical_name = canonical_forecasting_model_name(model_name)
    resolved_params = dict(params or {})

    trajectory_transform = {
        'kind': 'native_context_window',
        'window_size': resolved_params.get('window_size', resolved_params.get('patch_len')),
        'stride': resolved_params.get('stride'),
        'history_length': training_history_length,
    }

    forecast_head = {
        'head_type': canonical_name,
        'epochs': resolved_params.get('epochs'),
        'batch_size': resolved_params.get('batch_size'),
        'learning_rate': resolved_params.get('learning_rate'),
        'activation': resolved_params.get('activation'),
        'forecast_horizon': int(forecast_horizon),
        'device': resolved_params.get('device', 'cpu'),
    }

    model_specific_fields = {
        'patch_tst_model': ('patch_len', 'forecast_mode', 'use_amp'),
        'tcn_model': ('patch_len', 'kernel_size', 'num_filters', 'num_layers', 'dilation_base', 'dropout'),
        'deepar_model': ('cell_type', 'hidden_size', 'rnn_layers', 'expected_distribution', 'dropout'),
        'nbeats_model': (
            'n_stacks',
            'n_trend_blocks',
            'n_seasonality_blocks',
            'n_of_harmonics',
            'layers',
            'degree_of_polynomial',
        ),
    }
    for key in model_specific_fields.get(canonical_name, ()):
        forecast_head[key] = resolved_params.get(key)

    return {
        'model_family': 'neural_forecaster',
        'model_name': canonical_name,
        'trajectory_transform': trajectory_transform,
        'decomposition': {},
        'rank_truncation': {},
        'forecast_head': forecast_head,
    }


def build_neural_forecasting_input_data(series: np.ndarray, *, forecast_horizon: int):
    try:
        from fedot.core.data.data import InputData
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    except Exception:
        class InputData:  # type: ignore[no-redef]
            def __init__(self, idx, features, target, task, data_type):
                self.idx = idx
                self.features = features
                self.target = target
                self.task = task
                self.data_type = data_type

        DataTypesEnum = SimpleNamespace(ts='ts')

        class TaskTypesEnum:  # type: ignore[no-redef]
            ts_forecasting = 'ts_forecasting'

        class TsForecastingParams:  # type: ignore[no-redef]
            def __init__(self, forecast_length: int):
                self.forecast_length = int(forecast_length)

        class Task:  # type: ignore[no-redef]
            def __init__(self, task_type, task_params):
                self.task_type = task_type
                self.task_params = task_params

    normalized = np.asarray(series, dtype=float).reshape(-1)
    return InputData(
        idx=np.arange(len(normalized)),
        features=normalized,
        target=normalized,
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=int(forecast_horizon))),
        data_type=DataTypesEnum.ts,
    )


def normalize_neural_forecast_prediction(prediction: Any, forecast_horizon: int) -> np.ndarray:
    values = np.asarray(getattr(prediction, 'predict', prediction), dtype=float).reshape(-1)
    return values[:int(forecast_horizon)]


def build_neural_forecast_head(
        model_name: str,
        *,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
) -> 'NeuralForecastHead':
    return NeuralForecastHead(
        spec=NeuralForecastHeadSpec(
            model_name=model_name,
            forecast_horizon=int(forecast_horizon),
            params=params,
        )
    )


@dataclass
class NeuralForecastHead:
    spec: NeuralForecastHeadSpec

    def __post_init__(self):
        self.canonical_model_name = self.spec.model_name
        self.params = dict(self.spec.params)
        self.forecast_horizon = int(self.spec.forecast_horizon)
        self.model_cls_ = resolve_neural_forecasting_model_cls(self.canonical_model_name)
        self.model_ = None
        self.training_history_ = None
        self.diagnostics_ = {}

    def fit(self, time_series: np.ndarray):
        try:
            from fedot.core.operations.operation_parameters import OperationParameters
            model_params = OperationParameters(**self.params)
        except Exception:
            model_params = dict(self.params)

        history = np.asarray(time_series, dtype=float).reshape(-1)
        self.training_history_ = history
        self.model_ = self.model_cls_(model_params)
        self.model_.fit(build_neural_forecasting_input_data(history, forecast_horizon=self.forecast_horizon))
        self.diagnostics_ = build_neural_forecasting_stage_diagnostics(
            self.canonical_model_name,
            forecast_horizon=self.forecast_horizon,
            params=self.params,
            training_history_length=len(history),
        )
        return self

    def predict(self, time_series: np.ndarray | None = None, forecast_horizon: int | None = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError('NeuralForecastHead must be fitted before predict().')
        horizon = int(forecast_horizon or self.forecast_horizon)
        if horizon != self.forecast_horizon:
            raise ValueError(
                f'NeuralForecastHead was fitted for horizon={self.forecast_horizon}, got {horizon}.'
            )
        context = self.training_history_ if time_series is None else np.asarray(time_series, dtype=float).reshape(-1)
        raw_prediction = self.model_.predict(
            build_neural_forecasting_input_data(context, forecast_horizon=self.forecast_horizon)
        )
        values = normalize_neural_forecast_prediction(raw_prediction, self.forecast_horizon)
        self.diagnostics_ = {
            **dict(self.diagnostics_),
            'last_prediction_diagnostics': {
                'forecast_shape': tuple(int(value) for value in values.shape),
                'first_prediction_value': float(values[0]) if len(values) else None,
            },
        }
        return values

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostics_)
