"""Idempotent effect boundary for registering Industrial operations in FEDOT."""

from __future__ import annotations

from contextlib import contextmanager
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
    IndustrialExtensionStatus,
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

    from fedot.extensions import get_registered_extensions, register_extension
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
    if dry_run:
        return IndustrialExtensionResult(IndustrialExtensionStatus.PLANNED, plan)

    result = register_extension(manifest)
    if result.is_left():
        error = result.monoid[0]
        return _rejected(
            plan,
            IndustrialExtensionErrorCode.REGISTRATION_REJECTED,
            error.message,
            fedot_error_code=error.code,
            **error.details,
        )
    return IndustrialExtensionResult(IndustrialExtensionStatus.REGISTERED, plan)


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
    """Expose Industrial operations for one scope and always restore the registry."""
    from fedot.extensions import extension_scope, get_registered_extensions
    from fedot_ind.integration.fedot.extensions.manifest import build_industrial_extension_manifest

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

    with extension_scope(manifest):
        yield IndustrialExtensionResult(IndustrialExtensionStatus.REGISTERED, plan)


def _manifest_signature(manifest: object) -> tuple[object, ...]:
    def operation_signature(spec: object) -> tuple[object, ...]:
        schema = spec.hyperparams_schema
        capabilities = spec.capabilities
        return (
            spec.name,
            type(spec).__name__,
            capabilities,
            schema.required,
            schema.optional,
            tuple(sorted(schema.defaults.items())),
        )

    return (
        manifest.name,
        manifest.version,
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
