"""Deferred runtime factories for operations declared in the Industrial catalog."""

from __future__ import annotations

from importlib import import_module
import inspect
from typing import Any, Callable, Mapping

import numpy as np

from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
    IndustrialOperationDeclaration,
    IndustrialOperationKind,
    IndustrialRuntimeInterface,
)


def make_deferred_factory(declaration: IndustrialOperationDeclaration) -> Callable[..., object]:
    """Build a stable factory without importing the declared implementation."""
    wrapper_type = DeferredModel if declaration.kind is IndustrialOperationKind.MODEL else DeferredTransform

    def factory(params: Mapping[str, Any] | None = None) -> object:
        return wrapper_type(declaration, params)

    factory.__name__ = f"create_{declaration.name}"
    factory.__qualname__ = factory.__name__
    factory.__module__ = __name__
    factory.__industrial_factory_target__ = declaration.factory
    return factory


class _DeferredRuntime:
    """Load an Industrial implementation only when its runtime method is used."""

    def __init__(self, declaration: IndustrialOperationDeclaration,
                 params: Mapping[str, Any] | None = None) -> None:
        self.declaration = declaration
        self.params = dict(params or {})
        self._implementation: object | None = None
        self._task = None

    @property
    def implementation(self) -> object:
        if self._implementation is None:
            self._implementation = _instantiate_target(self.declaration, self.params)
        return self._implementation

    def _method(self, name: str) -> Callable[..., Any]:
        method = getattr(self.implementation, name, None)
        if not callable(method):
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
                "Industrial runtime target does not implement the required method.",
                context={"operation": self.declaration.name, "method": name},
            )
        return method


class DeferredModel(_DeferredRuntime):
    """Array-level model facade expected by the FEDOT extension runtime."""

    def fit(self, features: Any, target: Any = None) -> object:
        runtime_features = self._runtime_input(features, target)
        candidates = (
            ((runtime_features, target), (runtime_features,))
            if self.declaration.runtime_interface is IndustrialRuntimeInterface.ARRAY
            else ((runtime_features,),)
        )
        fitted = _invoke_supported(self._method("fit"), candidates, self.declaration.name)
        if fitted is not None and fitted is not self.implementation:
            self._implementation = fitted
        return self

    def predict(self, features: Any) -> Any:
        runtime_features = self._runtime_input(features)
        prediction = _unwrap_output(
            _invoke_supported(self._method("predict"), ((runtime_features,),), self.declaration.name),
            preferred=("predict", "features"),
        )
        return self._normalize_prediction_shape(prediction, features)

    def predict_proba(self, features: Any) -> Any:
        runtime_features = self._runtime_input(features)
        method = getattr(self.implementation, "predict_proba", None)
        candidates = ((runtime_features,),)
        if not callable(method):
            method = self._method("predict")
            candidates = ((runtime_features, "probs"),)
        return _unwrap_output(
            _invoke_supported(method, candidates, self.declaration.name),
            preferred=("predict", "features"),
        )

    def _runtime_input(self, features: Any, target: Any = None) -> Any:
        if self.declaration.runtime_interface is IndustrialRuntimeInterface.ARRAY:
            return features
        return _build_input_data(self, features, target)

    def _normalize_prediction_shape(self, prediction: Any, features: Any) -> Any:
        task_type = getattr(getattr(self._task, "task_type", None), "value", None)
        if task_type != "ts_forecasting":
            return prediction
        sample_count = len(features)
        if hasattr(prediction, "reshape"):
            return prediction.reshape(sample_count, -1)
        return np.asarray(prediction).reshape(sample_count, -1)


class DeferredTransform(_DeferredRuntime):
    """Array-level transform facade expected by the FEDOT extension runtime."""

    def fit(self, features: Any, target: Any = None) -> object:
        runtime_features = self._runtime_input(features, target)
        candidates = (
            ((runtime_features, target), (runtime_features,))
            if self.declaration.runtime_interface is IndustrialRuntimeInterface.ARRAY
            else ((runtime_features,),)
        )
        fitted = _invoke_supported(self._method("fit"), candidates, self.declaration.name)
        if fitted is not None and fitted is not self.implementation:
            self._implementation = fitted
        return self

    def transform(self, features: Any) -> Any:
        method = getattr(self.implementation, "transform", None)
        if not callable(method):
            method = self._method("predict")
        runtime_features = self._runtime_input(features)
        transformed = _unwrap_transform_output(
            _invoke_supported(method, ((runtime_features,),), self.declaration.name)
        )
        return _normalize_transform_shape(transformed, self.declaration)

    def _runtime_input(self, features: Any, target: Any = None) -> Any:
        if self.declaration.runtime_interface is IndustrialRuntimeInterface.ARRAY:
            return features
        return _build_input_data(self, features, target)


def _build_input_data(runtime: _DeferredRuntime, features: Any, target: Any = None) -> object:
    from fedot.core.data.input_data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    values = _as_numpy(features)
    target_values = None if target is None else _as_numpy(target)
    if runtime._task is None:
        task_name = _resolve_task(runtime.declaration, runtime.params)
        task_params = None
        if task_name == "ts_forecasting":
            raw_horizon = runtime.params.get("forecast_length")
            if raw_horizon is None and target_values is not None:
                target_shape = np.asarray(target_values).shape
                raw_horizon = target_shape[-1] if len(target_shape) > 1 else len(target_values)
            horizon = 1 if raw_horizon is None else int(raw_horizon)
            task_params = TsForecastingParams(forecast_length=max(1, horizon))
        runtime._task = Task(TaskTypesEnum[task_name], task_params)
    data_type_name = runtime.declaration.data_types[0]
    data_type = DataTypesEnum[data_type_name]
    return InputData(
        idx=np.arange(len(values)),
        features=values,
        target=target_values,
        task=runtime._task,
        data_type=data_type,
    )


def _resolve_task(
        declaration: IndustrialOperationDeclaration,
        params: Mapping[str, Any],
) -> str:
    configured_task = params.get("task_type")
    if configured_task is not None:
        task_name = str(getattr(configured_task, "value", configured_task))
        if task_name not in declaration.tasks:
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.RUNTIME_TASK_REQUIRED,
                "The configured task_type is not supported by the Industrial operation.",
                context={
                    "operation": declaration.name,
                    "task_type": task_name,
                    "supported_tasks": declaration.tasks,
                },
            )
        return task_name
    if len(declaration.tasks) == 1:
        return declaration.tasks[0]
    if not declaration.needs_explicit_task_type:
        return declaration.tasks[0]
    raise IndustrialExtensionContractError(
        IndustrialExtensionErrorCode.RUNTIME_TASK_REQUIRED,
        "A multi-task Industrial model requires an explicit task_type parameter.",
        context={"operation": declaration.name, "supported_tasks": declaration.tasks},
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _instantiate_target(declaration: IndustrialOperationDeclaration,
                        params: Mapping[str, Any]) -> object:
    module_name, attribute_name = declaration.factory.split(":", maxsplit=1)
    try:
        target = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.RUNTIME_TARGET_UNAVAILABLE,
            "Industrial runtime target cannot be imported.",
            context={"operation": declaration.name, "target": declaration.factory},
            cause=error,
        ) from error
    if not callable(target):
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
            "Industrial runtime target is not callable.",
            context={"operation": declaration.name, "target": declaration.factory},
        )

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
            "Industrial runtime target signature cannot be inspected.",
            context={"operation": declaration.name, "target": declaration.factory},
            cause=error,
        ) from error
    candidates = _constructor_candidates(signature, params)
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        try:
            return target(*args, **kwargs)
        except Exception as error:
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
                "Industrial runtime target construction failed.",
                context={"operation": declaration.name, "target": declaration.factory},
                cause=error,
            ) from error
    raise IndustrialExtensionContractError(
        IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
        "Industrial runtime target constructor has no supported call shape.",
        context={"operation": declaration.name, "target": declaration.factory,
                 "signature": str(signature)},
    )


def _constructor_candidates(
        signature: inspect.Signature,
        params: Mapping[str, Any],
) -> tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]:
    """Choose constructor forms without binding a parameter object to an unrelated first argument."""
    keyword_params = dict(params)
    if "params" not in signature.parameters:
        return (((), keyword_params), ((), {}))
    operation_params = _operation_parameters(params)
    return (
        ((operation_params,), {}),
        ((), {"params": operation_params}),
        ((), keyword_params),
        ((), {}),
    )


def _operation_parameters(params: Mapping[str, Any]) -> object:
    try:
        from fedot.core.operations.operation_parameters import OperationParameters
    except ImportError as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE,
            "FEDOT OperationParameters is unavailable.",
            cause=error,
        ) from error
    return OperationParameters(**dict(params))


def _invoke_supported(method: Callable[..., Any], candidates: tuple[tuple[Any, ...], ...],
                      operation_name: str) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
            "Industrial runtime method signature cannot be inspected.",
            context={"operation": operation_name},
            cause=error,
        ) from error
    for args in candidates:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        try:
            return method(*args)
        except Exception as error:
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
                "Industrial runtime method failed.",
                context={"operation": operation_name, "method": getattr(method, "__name__", repr(method))},
                cause=error,
            ) from error
    raise IndustrialExtensionContractError(
        IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
        "Industrial runtime method has no supported call shape.",
        context={"operation": operation_name, "signature": str(signature)},
    )


def _unwrap_output(value: Any, preferred: tuple[str, ...]) -> Any:
    for attribute in preferred:
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return candidate
    return value


def _unwrap_transform_output(value: Any) -> Any:
    try:
        from fedot.core.data.input_data.data import OutputData
    except ImportError:
        OutputData = ()
    if isinstance(value, OutputData):
        return value.predict
    return _unwrap_output(value, preferred=("features", "predict"))


def _normalize_transform_shape(
        value: Any,
        declaration: IndustrialOperationDeclaration,
) -> np.ndarray:
    array = _as_numpy(value)
    if declaration.output_data_type == "tabular" and array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    if array.ndim <= 3:
        return array
    if declaration.output_data_type == "image":
        return array.reshape(array.shape[0], -1, array.shape[-1])
    raise IndustrialExtensionContractError(
        IndustrialExtensionErrorCode.RUNTIME_TARGET_INVALID,
        "Industrial transform returned more than three feature axes.",
        context={
            "operation": declaration.name,
            "shape": array.shape,
            "output_data_type": declaration.output_data_type,
        },
    )
