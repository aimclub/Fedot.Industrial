"""Pure values used by the Industrial extension integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class IndustrialOperationKind(str, Enum):
    """Operation kinds supported by the FEDOT extension contract."""

    MODEL = "model"
    TRANSFORM = "transform"


class IndustrialRuntimeInterface(str, Enum):
    """Input convention implemented by an Industrial runtime target."""

    ARRAY = "array"
    INPUT_DATA = "input_data"


class IndustrialExtensionStatus(str, Enum):
    """Observable outcomes of the idempotent bootstrap operation."""

    PLANNED = "planned"
    REGISTERED = "registered"
    ALREADY_REGISTERED = "already_registered"
    REJECTED = "rejected"


class IndustrialExtensionSessionState(str, Enum):
    """Lifecycle of an owned FEDOT extension scope."""

    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"


class IndustrialExtensionErrorCode(str, Enum):
    """Stable errors produced before or around the FEDOT registry boundary."""

    CATALOG_NOT_FOUND = "catalog_not_found"
    INVALID_CATALOG = "invalid_catalog"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    DUPLICATE_OPERATION = "duplicate_operation"
    INVALID_OPERATION = "invalid_operation"
    FEDOT_CONTRACT_UNAVAILABLE = "fedot_contract_unavailable"
    REGISTRATION_CONFLICT = "registration_conflict"
    REGISTRATION_REJECTED = "registration_rejected"
    OPERATION_NOT_FOUND = "operation_not_found"
    RUNTIME_TARGET_UNAVAILABLE = "runtime_target_unavailable"
    RUNTIME_TARGET_INVALID = "runtime_target_invalid"
    RUNTIME_TASK_REQUIRED = "runtime_task_required"
    INVALID_SESSION_STATE = "invalid_session_state"


class IndustrialExtensionContractError(ValueError):
    """Expected integration failure with a serializable machine-readable code."""

    def __init__(
            self,
            code: IndustrialExtensionErrorCode,
            message: str,
            *,
            context: Mapping[str, Any] | None = None,
            cause: Exception | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.cause = cause
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": _jsonable(self.context),
        }


@dataclass(frozen=True)
class IndustrialOperationDeclaration:
    """Validated, FEDOT-independent operation declaration from the catalog."""

    name: str
    kind: IndustrialOperationKind
    factory: str
    tasks: tuple[str, ...]
    data_types: tuple[str, ...]
    output_data_type: str
    tags: tuple[str, ...]
    problems: tuple[str, ...]
    defaults_key: str | None = None
    supports_multimodal: bool = False
    requires_target: bool = True
    requires_fit: bool = True
    requires_task_type: bool = False
    backend: str = "numpy"
    description: str = ""
    runtime_interface: IndustrialRuntimeInterface = IndustrialRuntimeInterface.INPUT_DATA

    @property
    def needs_explicit_task_type(self) -> bool:
        return self.requires_task_type or (
            self.kind is IndustrialOperationKind.MODEL
            and self.runtime_interface is IndustrialRuntimeInterface.INPUT_DATA
            and len(self.tasks) > 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "factory": self.factory,
            "tasks": list(self.tasks),
            "data_types": list(self.data_types),
            "output_data_type": self.output_data_type,
            "tags": list(self.tags),
            "problems": list(self.problems),
            "defaults_key": self.defaults_key,
            "supports_multimodal": self.supports_multimodal,
            "requires_target": self.requires_target,
            "requires_fit": self.requires_fit,
            "requires_task_type": self.requires_task_type,
            "backend": self.backend,
            "description": self.description,
            "runtime_interface": self.runtime_interface.value,
        }


@dataclass(frozen=True)
class IndustrialExtensionCatalog:
    """Immutable source description used to construct a FEDOT manifest."""

    schema_version: int
    name: str
    version: str
    description: str
    operations: tuple[IndustrialOperationDeclaration, ...]
    digest: str

    def operation(self, name: str) -> IndustrialOperationDeclaration | None:
        return next((operation for operation in self.operations if operation.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
            },
            "operations": [operation.to_dict() for operation in self.operations],
            "digest": self.digest,
        }


@dataclass(frozen=True)
class IndustrialExtensionPlan:
    """Serializable decision prepared before mutating the FEDOT registry."""

    manifest_name: str
    manifest_version: str
    catalog_digest: str
    model_names: tuple[str, ...]
    transform_names: tuple[str, ...]

    @property
    def operation_names(self) -> tuple[str, ...]:
        return self.model_names + self.transform_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_name": self.manifest_name,
            "manifest_version": self.manifest_version,
            "catalog_digest": self.catalog_digest,
            "model_names": list(self.model_names),
            "transform_names": list(self.transform_names),
        }


@dataclass(frozen=True)
class IndustrialExtensionResult:
    """Result returned by dry-run and registration entrypoints."""

    status: IndustrialExtensionStatus
    plan: IndustrialExtensionPlan
    error_code: str | None = None
    error_message: str | None = None
    error_context: Mapping[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.status is not IndustrialExtensionStatus.REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "plan": self.plan.to_dict(),
            "error": None if self.error_code is None else {
                "code": self.error_code,
                "message": self.error_message,
                "context": _jsonable(dict(self.error_context or {})),
            },
        }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return repr(value)
