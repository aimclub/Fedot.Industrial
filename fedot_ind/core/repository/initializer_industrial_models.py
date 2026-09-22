"""Compatibility facade for activating Industrial operations in FEDOT."""

from __future__ import annotations

from importlib import import_module
import os
from typing import Any

from fedot_ind.integration.fedot.extensions.bootstrap import supports_fedot_extension_contract
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
)


PROFILE_ENVIRONMENT_VARIABLE = "FEDOT_INTEGRATION_PROFILE"
_ACTIVE_EXTENSION_SCOPE: Any | None = None
_ACTIVE_SCOPE_OWNER: Any | None = None
_ACTIVE_SCOPE_RESULT: Any | None = None


class IndustrialModels:
    """Activate the manifest contract or an explicitly selected legacy bridge."""

    def __init__(self, profile: str | None = None) -> None:
        selected = profile or os.getenv(PROFILE_ENVIRONMENT_VARIABLE)
        self.profile = selected or ("tensor" if supports_fedot_extension_contract() else "legacy")
        if self.profile not in {"legacy", "tensor"}:
            raise IndustrialExtensionContractError(
                IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE,
                "Unknown FEDOT Industrial repository profile.",
                context={"profile": self.profile, "allowed": ["legacy", "tensor"]},
            )
        self._legacy: Any | None = None
        self.last_result: Any | None = None
        self._owns_context_scope = False

    def setup_repository(self, backend: str = "default"):
        """Activate Industrial operations without replacing FEDOT repositories."""
        if self.profile == "legacy":
            return self._legacy_repository().setup_repository(backend)
        self.last_result = _open_extension_scope(self)
        return _operation_repository()

    def setup_default_repository(self, backend: str = "default"):
        """Restore the state that preceded this facade activation."""
        if self.profile == "legacy":
            return self._legacy_repository().setup_default_repository(backend)
        _close_extension_scope(None, None, None)
        return _operation_repository()

    def __enter__(self):
        if self.profile == "legacy":
            self._legacy_repository().__enter__()
            return self
        self._owns_context_scope = _ACTIVE_EXTENSION_SCOPE is None
        self.last_result = _open_extension_scope(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.profile == "legacy":
            self._legacy_repository().__exit__(exc_type, exc_val, exc_tb)
            return
        if self._owns_context_scope:
            _close_extension_scope(exc_type, exc_val, exc_tb)
        self._owns_context_scope = False

    def _legacy_repository(self):
        if self._legacy is None:
            try:
                module = import_module("fedot_ind.integration.fedot.legacy_repository")
            except ImportError as error:
                raise IndustrialExtensionContractError(
                    IndustrialExtensionErrorCode.FEDOT_CONTRACT_UNAVAILABLE,
                    "The legacy FEDOT repository bridge is unavailable in this environment.",
                    context={"profile": self.profile},
                    cause=error,
                ) from error
            self._legacy = module.LegacyIndustrialRepository()
        return self._legacy


def _operation_repository():
    from fedot.core.repository.operation_types_repository import OperationTypesRepository

    return OperationTypesRepository


def _open_extension_scope(owner: IndustrialModels):
    global _ACTIVE_EXTENSION_SCOPE, _ACTIVE_SCOPE_OWNER, _ACTIVE_SCOPE_RESULT

    if _ACTIVE_EXTENSION_SCOPE is None:
        from fedot_ind.integration.fedot.extensions.bootstrap import industrial_extension_scope

        scope = industrial_extension_scope()
        result = scope.__enter__()
        _ACTIVE_EXTENSION_SCOPE = scope
        _ACTIVE_SCOPE_OWNER = owner
        _ACTIVE_SCOPE_RESULT = result
    return _ACTIVE_SCOPE_RESULT


def _close_extension_scope(exc_type, exc_val, exc_tb) -> None:
    global _ACTIVE_EXTENSION_SCOPE, _ACTIVE_SCOPE_OWNER, _ACTIVE_SCOPE_RESULT

    if _ACTIVE_EXTENSION_SCOPE is None:
        return
    scope = _ACTIVE_EXTENSION_SCOPE
    _ACTIVE_EXTENSION_SCOPE = None
    _ACTIVE_SCOPE_OWNER = None
    _ACTIVE_SCOPE_RESULT = None
    scope.__exit__(exc_type, exc_val, exc_tb)
