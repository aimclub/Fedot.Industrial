"""Deferred runtime factories for operations declared in the Industrial catalog."""

from __future__ import annotations

from importlib import import_module
import inspect
from typing import Any, Callable, Mapping

from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
    IndustrialOperationDeclaration,
    IndustrialOperationKind,
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
        fitted = _invoke_supported(self._method("fit"), ((features, target), (features,)), self.declaration.name)
        if fitted is not None and fitted is not self.implementation:
            self._implementation = fitted
        return self

    def predict(self, features: Any) -> Any:
        return _unwrap_output(
            _invoke_supported(self._method("predict"), ((features,),), self.declaration.name),
            preferred=("predict", "features"),
        )

    def predict_proba(self, features: Any) -> Any:
        return _unwrap_output(
            _invoke_supported(self._method("predict_proba"), ((features,),), self.declaration.name),
            preferred=("predict", "features"),
        )


class DeferredTransform(_DeferredRuntime):
    """Array-level transform facade expected by the FEDOT extension runtime."""

    def fit(self, features: Any, target: Any = None) -> object:
        fitted = _invoke_supported(self._method("fit"), ((features, target), (features,)), self.declaration.name)
        if fitted is not None and fitted is not self.implementation:
            self._implementation = fitted
        return self

    def transform(self, features: Any) -> Any:
        method = getattr(self.implementation, "transform", None)
        if not callable(method):
            method = self._method("predict")
        return _unwrap_output(
            _invoke_supported(method, ((features,),), self.declaration.name),
            preferred=("features", "predict"),
        )


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

    operation_params = _operation_parameters(params)
    candidates = (
        ((operation_params,), {}),
        ((), {"params": operation_params}),
        ((), dict(params)),
        ((), {}),
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
