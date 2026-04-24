from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from fedot_ind.core.models.nn.network_impl.forecasting_model.common import (
    DEFAULT_FORECASTING_NN_BATCH_SIZE,
    DEFAULT_FORECASTING_NN_DEVICE,
    DEFAULT_FORECASTING_NN_EPOCHS,
    DEFAULT_FORECASTING_NN_LEARNING_RATE,
    normalize_neural_forecasting_params,
)
from fedot_ind.core.repository.forecasting_registry import canonical_forecasting_model_name

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
        object.__setattr__(self, 'params', normalize_neural_forecasting_params(self.params))

    @property
    def family(self) -> str:
        return 'neural_forecaster'


@dataclass(frozen=True)
class NeuralForecastHeadRunResult:
    spec: NeuralForecastHeadSpec
    forecast: tuple[float, ...]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'spec': {
                'model_name': self.spec.model_name,
                'forecast_horizon': int(self.spec.forecast_horizon),
                'params': dict(self.spec.params),
                'family': self.spec.family,
            },
            'forecast': list(self.forecast),
            'diagnostics': dict(self.diagnostics),
            'metadata': dict(self.metadata),
        }


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
    resolved_params = normalize_neural_forecasting_params(params)

    trajectory_transform = {
        'kind': 'native_context_window',
        'window_size': resolved_params.get('window_size', resolved_params.get('patch_len')),
        'stride': resolved_params.get('stride'),
        'history_length': training_history_length,
    }

    forecast_head = {
        'head_type': canonical_name,
        'epochs': resolved_params.get('epochs', DEFAULT_FORECASTING_NN_EPOCHS),
        'batch_size': resolved_params.get('batch_size', DEFAULT_FORECASTING_NN_BATCH_SIZE),
        'learning_rate': resolved_params.get('learning_rate', DEFAULT_FORECASTING_NN_LEARNING_RATE),
        'activation': resolved_params.get('activation'),
        'forecast_horizon': int(forecast_horizon),
        'device': resolved_params.get('device', DEFAULT_FORECASTING_NN_DEVICE),
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping': 'enabled',
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


def run_neural_forecast_head_on_series(
        model_name: str,
        *,
        time_series: np.ndarray,
        forecast_horizon: int,
        params: dict[str, Any] | None = None,
) -> NeuralForecastHeadRunResult:
    spec = NeuralForecastHeadSpec(
        model_name=model_name,
        forecast_horizon=int(forecast_horizon),
        params=params,
    )
    head = NeuralForecastHead(spec=spec)
    history = np.asarray(time_series, dtype=float).reshape(-1)
    head.fit(history)
    forecast = head.predict(history)
    return NeuralForecastHeadRunResult(
        spec=spec,
        forecast=tuple(float(value) for value in np.asarray(forecast, dtype=float).reshape(-1).tolist()),
        diagnostics=head.get_diagnostics(),
        metadata={
            'history_length': int(len(history)),
            'model_family': spec.family,
        },
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


class NeuralForecastHeadImplementation(ModelImplementation):
    model_name: str = ''

    def __init__(self, params: OperationParameters | None = None):
        params = params or OperationParameters()
        super().__init__(params)
        self.model_: NeuralForecastHead | None = None

    def fit(self, input_data: InputData):
        self.model_ = build_neural_forecast_head(
            self.model_name,
            forecast_horizon=int(input_data.task.task_params.forecast_length),
            params=dict(self.params),
        )
        self.model_.fit(np.asarray(input_data.features, dtype=float))
        return self

    def predict(self, input_data: InputData) -> OutputData:
        prediction = self.model_.predict(np.asarray(input_data.features, dtype=float))
        return self._convert_to_output(
            input_data,
            predict=np.asarray(prediction, dtype=float),
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


class PatchTSTForecastHeadImplementation(NeuralForecastHeadImplementation):
    model_name = 'patch_tst_model'


class TCNForecastHeadImplementation(NeuralForecastHeadImplementation):
    model_name = 'tcn_model'


class DeepARForecastHeadImplementation(NeuralForecastHeadImplementation):
    model_name = 'deepar_model'


class NBeatsForecastHeadImplementation(NeuralForecastHeadImplementation):
    model_name = 'nbeats_model'
