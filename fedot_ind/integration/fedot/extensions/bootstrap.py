"""Idempotent effect boundary for registering Industrial operations in FEDOT."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib.util import find_spec
from typing import Iterator

from pymonad.either import Left, Right

from fedot_ind.integration.fedot.extensions.catalog import (
    build_industrial_extension_plan,
    load_industrial_extension_catalog,
)
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
    IndustrialExtensionResult,
    IndustrialExtensionSessionState,
    IndustrialExtensionStatus,
)


@dataclass(frozen=True)
class _OwnedExtensionScope:
    scope: object
    result: IndustrialExtensionResult
    owners: int


_OWNED_EXTENSION_SCOPE: ContextVar[_OwnedExtensionScope | None] = ContextVar(
    "industrial_owned_extension_scope",
    default=None,
)


def supports_fedot_extension_contract() -> bool:
    """Report whether the active FEDOT installation exposes the extension API."""
    try:
        return find_spec("fedot.extensions") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def register_industrial_extension(*, dry_run: bool = False) -> IndustrialExtensionResult:
    """Register the Industrial manifest once or return a structured conflict."""
    plan = build_industrial_extension_plan()
    if not supports_fedot_extension_contract():
        return _rejected(
            plan,
            IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE,
            "The active FEDOT installation has no extension contract.",
        )

    from fedot.extensions import get_registered_extensions, register_extension, register_extensions
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest

    manifest = build_industrial_extension_manifest()
    current = tuple(item.manifest for item in get_registered_extensions()
                    if item.manifest.name == manifest.name)
    if current:
        if len(current) == 1 and _manifest_signature(current[0]) == _manifest_signature(manifest):
            return IndustrialExtensionResult(IndustrialExtensionStatus.ALREADY_REGISTERED, plan)
        return _rejected(
            plan,
            IndustrialExtensionErrorCode.REGISTRATION_CONFLICT,
            "A different Industrial extension manifest is already registered.",
            registered_versions=[item.version for item in current],
        )
    result = (register_extensions((manifest,), dry_run=True)
              if dry_run else register_extension(manifest))
    if result.is_left():
        error = result.monoid[0]
        return _rejected(
            plan,
            IndustrialExtensionErrorCode.REGISTRATION_REJECTED,
            error.message,
            fedot_error_code=error.code,
            **error.details,
        )
    status = IndustrialExtensionStatus.PLANNED if dry_run else IndustrialExtensionStatus.REGISTERED
    return IndustrialExtensionResult(status, plan)


def require_industrial_extension() -> IndustrialExtensionResult:
    """Register Industrial operations or raise their structured failure."""
    result = register_industrial_extension()
    if result.is_success:
        return result
    code = IndustrialExtensionErrorCode(result.error_code)
    raise IndustrialExtensionContractError(
        code,
        result.error_message or "FEDOT rejected the Industrial extension.",
        context=dict(result.error_context or {}),
    )


class IndustrialExtensionSession:
    """Own one explicit extension scope for a long-lived Industrial API."""

    def __init__(self) -> None:
        self._scope = None
        self._result: IndustrialExtensionResult | None = None
        self._state = IndustrialExtensionSessionState.CREATED

    @property
    def state(self) -> IndustrialExtensionSessionState:
        return self._state

    @property
    def result(self) -> IndustrialExtensionResult | None:
        return self._result

    def activate(self) -> IndustrialExtensionResult:
        if self._state is IndustrialExtensionSessionState.ACTIVE:
            assert self._result is not None
            return self._result
        if self._state is IndustrialExtensionSessionState.CLOSED:
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.INVALID_SESSION_STATE,
                "A closed Industrial extension session cannot be activated again.",
                context={"state": self._state.value},
            )
        scope = industrial_extension_scope()
        result = scope.__enter__()
        self._scope = scope
        self._result = result
        self._state = IndustrialExtensionSessionState.ACTIVE
        return result

    def close(self) -> None:
        if self._state is IndustrialExtensionSessionState.CLOSED:
            return
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
        self._scope = None
        self._result = None
        self._state = IndustrialExtensionSessionState.CLOSED

    def __enter__(self) -> "IndustrialExtensionSession":
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._scope is not None:
            self._scope.__exit__(exc_type, exc_val, exc_tb)
            self._scope = None
            self._result = None
        self._state = IndustrialExtensionSessionState.CLOSED


def resolve_industrial_operation(operation_name: str):
    """Resolve a catalog declaration without mutating the FEDOT registry."""
    operation = load_industrial_extension_catalog().operation(operation_name.split("/", maxsplit=1)[0])
    if operation is None:
        return Left(IndustrialExtensionContractError(
            IndustrialExtensionErrorCode.OPERATION_NOT_FOUND,
            "Industrial operation is not declared in the extension catalog.",
            context={"operation": operation_name},
        ))
    return Right(operation)


@contextmanager
def industrial_extension_scope() -> Iterator[IndustrialExtensionResult]:
    """Expose Industrial operations while at least one local owner remains active."""
    from fedot.extensions import extension_scope, get_registered_extensions
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest

    owned = _OWNED_EXTENSION_SCOPE.get()
    if owned is not None:
        _OWNED_EXTENSION_SCOPE.set(replace(owned, owners=owned.owners + 1))
        try:
            yield IndustrialExtensionResult(
                IndustrialExtensionStatus.ALREADY_REGISTERED,
                owned.result.plan,
            )
        finally:
            _release_owned_scope(owned.scope)
        return

    manifest = build_industrial_extension_manifest()
    plan = build_industrial_extension_plan()
    current = tuple(item.manifest for item in get_registered_extensions()
                    if item.manifest.name == manifest.name)
    if current:
        if len(current) != 1 or _manifest_signature(current[0]) != _manifest_signature(manifest):
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.REGISTRATION_CONFLICT,
                "A different Industrial extension manifest is already registered.",
                context={"manifest": manifest.name},
            )
        yield IndustrialExtensionResult(IndustrialExtensionStatus.ALREADY_REGISTERED, plan)
        return

    scope = extension_scope(manifest)
    scope.__enter__()
    result = IndustrialExtensionResult(IndustrialExtensionStatus.REGISTERED, plan)
    _OWNED_EXTENSION_SCOPE.set(_OwnedExtensionScope(scope, result, 1))
    try:
        yield result
    except BaseException as error:
        _release_owned_scope(scope, type(error), error, error.__traceback__)
        raise
    else:
        _release_owned_scope(scope)


def _release_owned_scope(scope: object, exc_type=None, exc_val=None, exc_tb=None) -> None:
    owned = _OWNED_EXTENSION_SCOPE.get()
    if owned is None or owned.scope is not scope:
        return
    if owned.owners > 1:
        _OWNED_EXTENSION_SCOPE.set(replace(owned, owners=owned.owners - 1))
        return
    _OWNED_EXTENSION_SCOPE.set(None)
    scope.__exit__(exc_type, exc_val, exc_tb)


def _manifest_signature(manifest: object) -> tuple[object, ...]:
    def operation_signature(spec: object) -> tuple[object, ...]:
        schema = spec.hyperparams_schema
        capabilities = spec.capabilities
        return (
            spec.name,
            type(spec).__name__,
            getattr(spec.factory, "__industrial_factory_target__", None),
            getattr(spec.factory, "__industrial_invocation_policy__", None),
            getattr(spec.factory, "__module__", None),
            getattr(spec.factory, "__qualname__", None),
            capabilities,
            schema.required,
            schema.optional,
            tuple(sorted(schema.defaults.items())),
        )

    return (
        manifest.name,
        manifest.version,
        manifest.description,
        tuple(operation_signature(spec) for spec in manifest.models),
        tuple(operation_signature(spec) for spec in manifest.transforms),
    )


def _rejected(plan: object, code: IndustrialExtensionErrorCode,
              message: str, **context: object) -> IndustrialExtensionResult:
    return IndustrialExtensionResult(
        IndustrialExtensionStatus.REJECTED,
        plan,
        error_code=code.value,
        error_message=message,
        error_context=context,
    )
