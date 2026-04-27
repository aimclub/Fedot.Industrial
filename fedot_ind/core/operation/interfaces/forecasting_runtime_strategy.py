import warnings
from inspect import signature
from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.time_series import FedotTsForecastingStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.random import ImplementationRandomStateHandler

from fedot_ind.core.repository.forecasting_registry import CANONICAL_STAGE_FORECASTING_MODELS
from fedot_ind.core.repository.model_repository import FORECASTING_MODELS, FORECASTING_PREPROC

RUNTIME_FORECASTING_MODELS = set(CANONICAL_STAGE_FORECASTING_MODELS) | {
    'eigen_forecaster',
    'glm',
    'lagged_forecaster',
    'patch_tst_model',
    'tst_model',
    'deepar_model',
    'tcn_model',
    'nbeats_model',
}


def is_forecasting_model_operation(operation_type: str) -> bool:
    return str(operation_type) in FORECASTING_MODELS


def is_forecasting_preprocessing_operation(operation_type: str) -> bool:
    return str(operation_type) in FORECASTING_PREPROC


def should_redirect_legacy_model_strategy(strategy_cls: type, operation_type: str) -> bool:
    """Centralized redirect policy for legacy model strategies."""
    return strategy_cls.__name__ in {
        'IndustrialSkLearnEvaluationStrategy',
        'IndustrialSkLearnRegressionStrategy',
        'IndustrialCustomRegressionStrategy',
    } and is_forecasting_model_operation(operation_type)


def should_redirect_legacy_preprocessing_strategy(strategy_cls: type, operation_type: str) -> bool:
    """Centralized redirect policy for legacy preprocessing strategies."""
    return strategy_cls.__name__ in {
        'IndustrialCustomPreprocessingStrategy',
        'IndustrialPreprocessingStrategy',
    } and is_forecasting_preprocessing_operation(operation_type)


class LegacyForecastingModelRedirectMixin:
    """Minimal legacy boundary that redirects forecasting models into the dedicated runtime."""

    def __new__(cls, operation_type: str, params: Optional[OperationParameters] = None):
        if should_redirect_legacy_model_strategy(cls, operation_type):
            return IndustrialForecastingModelRuntimeStrategy(operation_type, params=params)
        return object.__new__(cls)


class LegacyForecastingPreprocessingRedirectMixin:
    """Minimal legacy boundary that redirects forecasting preprocessing into the dedicated runtime."""

    def __new__(cls, operation_type: str, params: Optional[OperationParameters] = None):
        if should_redirect_legacy_preprocessing_strategy(cls, operation_type):
            return IndustrialForecastingPreprocessingRuntimeStrategy(operation_type, params=params)
        return object.__new__(cls)


def _supports_output_mode(method) -> bool:
    try:
        return 'output_mode' in signature(method).parameters
    except (TypeError, ValueError):
        return False


def _invoke_with_optional_output_mode(method, input_data: InputData, output_mode: str):
    if _supports_output_mode(method):
        return method(input_data, output_mode=output_mode)
    return method(input_data)


def _invoke_strategy_with_optional_output_mode(method, trained_operation, input_data: InputData, output_mode: str):
    if _supports_output_mode(method):
        return method(trained_operation, input_data, output_mode=output_mode)
    return method(trained_operation, input_data)


def _normalize_legacy_ts_input(input_data: InputData) -> InputData:
    features = np.asarray(input_data.features)
    if features.ndim <= 1 or 1 not in features.shape:
        return input_data
    normalized_features = features.reshape(-1)
    idx = input_data.idx if len(input_data.idx) == len(normalized_features) else np.arange(len(normalized_features))
    target = input_data.target if len(input_data.target) == len(normalized_features) else normalized_features
    return InputData(
        idx=idx,
        features=normalized_features,
        target=target,
        task=input_data.task,
        data_type=input_data.data_type,
        supplementary_data=input_data.supplementary_data,
    )


class _BaseIndustrialForecastingStrategy(EvaluationStrategy):
    _operations_by_types = {}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.output_mode = self.params_for_fit.get('output_mode', 'default')

    def _instantiate(self):
        return self.operation_impl(self.params_for_fit)


class IndustrialForecastingModelRuntimeStrategy(_BaseIndustrialForecastingStrategy):
    """Thin forecasting-only model strategy without legacy multidim dispatch."""

    _operations_by_types = FORECASTING_MODELS

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self._use_legacy_fedot_strategy = str(operation_type) not in RUNTIME_FORECASTING_MODELS
        super().__init__(operation_type, params)
        self._legacy_strategy = (
            FedotTsForecastingStrategy(operation_type, params=params)
            if self._use_legacy_fedot_strategy else None
        )

    def fit(self, train_data: InputData):
        if self._legacy_strategy is not None:
            train_data = _normalize_legacy_ts_input(train_data)
            return self._legacy_strategy.fit(train_data)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self._instantiate()
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        if self._legacy_strategy is not None:
            predict_data = _normalize_legacy_ts_input(predict_data)
            return _invoke_strategy_with_optional_output_mode(
                self._legacy_strategy.predict,
                trained_operation,
                predict_data,
                output_mode,
            )
        prediction = _invoke_with_optional_output_mode(
            trained_operation.predict,
            predict_data,
            output_mode=output_mode,
        )
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        if self._legacy_strategy is not None:
            predict_data = _normalize_legacy_ts_input(predict_data)
            return _invoke_strategy_with_optional_output_mode(
                self._legacy_strategy.predict_for_fit,
                trained_operation,
                predict_data,
                output_mode,
            )
        if hasattr(trained_operation, 'predict_for_fit'):
            prediction = _invoke_with_optional_output_mode(
                trained_operation.predict_for_fit,
                predict_data,
                output_mode=output_mode,
            )
        else:
            prediction = _invoke_with_optional_output_mode(
                trained_operation.predict,
                predict_data,
                output_mode=output_mode,
            )
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)


class IndustrialForecastingPreprocessingRuntimeStrategy(_BaseIndustrialForecastingStrategy):
    """Thin forecasting-only preprocessing strategy without legacy multidim dispatch."""

    _operations_by_types = FORECASTING_PREPROC

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self._instantiate()
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData, output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.transform(predict_data)
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)

    def predict_for_fit(self, trained_operation, predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        transform_method = trained_operation.transform_for_fit \
            if hasattr(trained_operation, 'transform_for_fit') else trained_operation.transform
        prediction = transform_method(predict_data)
        return self._convert_to_output(prediction, predict_data, output_data_type=DataTypesEnum.table)
