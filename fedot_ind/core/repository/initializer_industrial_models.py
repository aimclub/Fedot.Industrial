"""Compatibility facade for activating Industrial operations in FEDOT."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from importlib import import_module
import os
from typing import Any

from fedot_ind.integration.fedot.extensions.bootstrap import supports_fedot_extension_contract
from fedot_ind.integration.fedot.extensions.contracts import (
    IndustrialExtensionContractError,
    IndustrialExtensionErrorCode,
)


PROFILE_ENVIRONMENT_VARIABLE = "FEDOT_INTEGRATION_PROFILE"


@dataclass(frozen=True)
class _ExtensionActivation:
    scope: Any
    result: Any
    manual_active: bool = False
    context_depth: int = 0


_ACTIVE_EXTENSION_STATE: ContextVar[_ExtensionActivation | None] = ContextVar(
    "fedot_industrial_extension_activation",
    default=None,
)


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

    def setup_repository(self, backend: str = "default"):
        """Activate Industrial operations without replacing FEDOT repositories."""
        if self.profile == "legacy":
            return self._legacy_repository().setup_repository(backend)
        self.last_result = _open_extension_scope(manual=True)
        return _operation_repository()

    def setup_default_repository(self, backend: str = "default"):
        """Restore the state that preceded this facade activation."""
        if self.profile == "legacy":
            return self._legacy_repository().setup_default_repository(backend)
        _close_extension_scope(None, None, None, manual=True)
        return _operation_repository()

    def __enter__(self):
        if self.profile == "legacy":
            self._legacy_repository().__enter__()
            return self
        self.last_result = _open_extension_scope(context=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.profile == "legacy":
            self._legacy_repository().__exit__(exc_type, exc_val, exc_tb)
            return
        _close_extension_scope(exc_type, exc_val, exc_tb, context=True)

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


def _open_extension_scope(*, manual: bool = False, context: bool = False):
    state = _ACTIVE_EXTENSION_STATE.get()
    if state is None:
        from fedot_ind.integration.fedot.extensions.bootstrap import industrial_extension_scope

        scope = industrial_extension_scope()
        result = scope.__enter__()
        state = _ExtensionActivation(scope=scope, result=result)
    state = replace(
        state,
        manual_active=state.manual_active or manual,
        context_depth=state.context_depth + int(context),
    )
    _ACTIVE_EXTENSION_STATE.set(state)
    return state.result


def _close_extension_scope(exc_type, exc_val, exc_tb, *,
                           manual: bool = False, context: bool = False) -> None:
    state = _ACTIVE_EXTENSION_STATE.get()
    if state is None:
        return
    next_state = replace(
        state,
        manual_active=False if manual else state.manual_active,
        context_depth=max(0, state.context_depth - int(context)),
    )
    if next_state.manual_active or next_state.context_depth:
        _ACTIVE_EXTENSION_STATE.set(next_state)
        return
    _ACTIVE_EXTENSION_STATE.set(None)
    state.scope.__exit__(exc_type, exc_val, exc_tb)
