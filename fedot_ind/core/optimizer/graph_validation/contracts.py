"""Immutable contracts for capability-aware graph validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


GRAPH_SPEC_SCHEMA = "industrial_graph_spec@1"
GRAPH_VALIDATION_SCHEMA = "industrial_graph_validation@1"


class OperationKind(str, Enum):
    """Graph-level operation roles relevant to validation."""

    MODEL = "model"
    TRANSFORM = "transform"
    DATA_SOURCE = "data_source"
    UNKNOWN = "unknown"


class ComputeDevice(str, Enum):
    """Execution devices declared by an operation capability."""

    CPU = "cpu"
    CUDA = "cuda"


class NodePosition(str, Enum):
    """Positions an operation may occupy in a directed pipeline."""

    ANY = "any"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ROOT = "root"


class ValidationSeverity(str, Enum):
    """Severity of one graph validation issue."""

    ERROR = "error"
    WARNING = "warning"


class ValidationIssueCode(str, Enum):
    """Stable machine-readable graph rejection reasons."""

    EMPTY_GRAPH = "empty_graph"
    DUPLICATE_NODE_ID = "duplicate_node_id"
    UNKNOWN_PARENT = "unknown_parent"
    SELF_CYCLE = "self_cycle"
    CYCLE = "cycle"
    ROOT_COUNT = "root_count"
    ISOLATED_NODE = "isolated_node"
    DISCONNECTED_COMPONENT = "disconnected_component"
    MISSING_PRIMARY_NODE = "missing_primary_node"
    UNKNOWN_OPERATION = "unknown_operation"
    TASK_NOT_SUPPORTED = "task_not_supported"
    DEVICE_NOT_SUPPORTED = "device_not_supported"
    OPERATION_NOT_SERIALIZABLE = "operation_not_serializable"
    POSITION_NOT_ALLOWED = "position_not_allowed"
    TOO_FEW_PARENTS = "too_few_parents"
    TOO_MANY_PARENTS = "too_many_parents"
    ROOT_NOT_MODEL = "root_not_model"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    IDENTICAL_PARENT_OPERATIONS = "identical_parent_operations"
    MIXED_DATA_SOURCES = "mixed_data_sources"
    RESAMPLE_POSITION = "resample_position"
    RESAMPLE_PARENT_CONFLICT = "resample_parent_conflict"
    DECOMPOSE_PARENT_COUNT = "decompose_parent_count"
    DECOMPOSE_MODEL_PARENT_REQUIRED = "decompose_model_parent_required"
    PARALLEL_FILTERING_BRANCHES = "parallel_filtering_branches"
    LEGACY_VERIFIER_REJECTED = "legacy_verifier_rejected"
    LEGACY_VERIFIER_FAILED = "legacy_verifier_failed"
    GRAPH_ADAPTATION_FAILED = "graph_adaptation_failed"


def _unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _freeze_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
               for item in value):
            return {item[0]: _json_value(item[1]) for item in value}
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class OperationCapabilities:
    """Validation-relevant operation contract independent of FEDOT runtime objects."""

    name: str
    kind: OperationKind
    input_data_types: tuple[str, ...]
    output_data_types: tuple[str, ...]
    task_types: tuple[str, ...]
    devices: tuple[ComputeDevice, ...] = (ComputeDevice.CPU,)
    serializable: bool = True
    allowed_positions: tuple[NodePosition, ...] = (NodePosition.ANY,)
    min_parents: int = 0
    max_parents: int | None = None
    allows_identical_parents: bool = True
    supports_multimodal: bool = False
    requires_target: bool = True
    requires_fit: bool = True
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Operation capability name must not be empty")
        if not isinstance(self.kind, OperationKind):
            raise TypeError("Operation capability kind must be an OperationKind")
        tuple_fields = (
            self.input_data_types,
            self.output_data_types,
            self.task_types,
            self.devices,
            self.allowed_positions,
            self.tags,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise TypeError("Operation capability collections must be tuples")
        if (isinstance(self.min_parents, bool)
                or not isinstance(self.min_parents, int)
                or self.min_parents < 0):
            raise ValueError("min_parents must be a non-negative integer")
        if self.max_parents is not None:
            if (isinstance(self.max_parents, bool)
                    or not isinstance(self.max_parents, int)
                    or self.max_parents < 0):
                raise ValueError("max_parents must be a non-negative integer or None")
            if self.max_parents < self.min_parents:
                raise ValueError("max_parents must not be smaller than min_parents")
        if (not self.devices
                or any(not isinstance(device, ComputeDevice) for device in self.devices)
                or len(self.devices) != len(set(self.devices))):
            raise ValueError("Operation capabilities must declare at least one device")
        if (not self.allowed_positions
                or any(not isinstance(position, NodePosition) for position in self.allowed_positions)
                or len(self.allowed_positions) != len(set(self.allowed_positions))):
            raise ValueError("Operation capabilities must declare unique valid positions")
        if NodePosition.ANY in self.allowed_positions and len(self.allowed_positions) > 1:
            raise ValueError("The 'any' position cannot be combined with explicit positions")
        flags = (
            self.serializable,
            self.allows_identical_parents,
            self.supports_multimodal,
            self.requires_target,
            self.requires_fit,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("Operation capability flags must be boolean")

    def supports_position(self, position: NodePosition) -> bool:
        return NodePosition.ANY in self.allowed_positions or position in self.allowed_positions

    def supports_any_position(self, positions: Iterable[NodePosition]) -> bool:
        return (NodePosition.ANY in self.allowed_positions
                or any(position in self.allowed_positions for position in positions))

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "input_data_types": list(self.input_data_types),
            "output_data_types": list(self.output_data_types),
            "task_types": list(self.task_types),
            "devices": [device.value for device in self.devices],
            "serializable": self.serializable,
            "allowed_positions": [position.value for position in self.allowed_positions],
            "min_parents": self.min_parents,
            "max_parents": self.max_parents,
            "allows_identical_parents": self.allows_identical_parents,
            "supports_multimodal": self.supports_multimodal,
            "requires_target": self.requires_target,
            "requires_fit": self.requires_fit,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class GraphNodeSpec:
    """One node and its incoming edges in an immutable graph specification."""

    node_id: str
    operation_name: str
    parent_ids: tuple[str, ...] = ()
    capabilities: OperationCapabilities | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("Graph node id must not be empty")
        if not isinstance(self.operation_name, str) or not self.operation_name.strip():
            raise ValueError("Graph operation name must not be empty")
        if not isinstance(self.parent_ids, tuple):
            raise TypeError("Graph parent identifiers must be a tuple")
        if any(not isinstance(parent_id, str) or not parent_id.strip()
               for parent_id in self.parent_ids):
            raise ValueError("Graph parent identifiers must be non-empty strings")

    def to_record(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "operation_name": self.operation_name,
            "parent_ids": list(self.parent_ids),
            "capabilities": None if self.capabilities is None else self.capabilities.to_record(),
        }


@dataclass(frozen=True)
class GraphSpec:
    """Effect-free input for graph validation."""

    graph_id: str
    nodes: tuple[GraphNodeSpec, ...]
    task_type: str | None = None
    requested_device: ComputeDevice | None = None
    require_serializable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("Graph id must not be empty")
        if self.task_type is not None and (
                not isinstance(self.task_type, str) or not self.task_type.strip()):
            raise ValueError("Graph task type must be a non-empty string or None")
        if self.requested_device is not None and not isinstance(
                self.requested_device, ComputeDevice):
            raise TypeError("requested_device must be a ComputeDevice or None")
        if not isinstance(self.require_serializable, bool):
            raise TypeError("require_serializable must be boolean")
        if not isinstance(self.nodes, tuple) or any(
                not isinstance(node, GraphNodeSpec) for node in self.nodes):
            raise TypeError("Graph nodes must be a tuple of GraphNodeSpec values")

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": GRAPH_SPEC_SCHEMA,
            "graph_id": self.graph_id,
            "task_type": self.task_type,
            "requested_device": None if self.requested_device is None else self.requested_device.value,
            "require_serializable": self.require_serializable,
            "nodes": [node.to_record() for node in self.nodes],
        }


@dataclass(frozen=True)
class ValidationIssue:
    """One precise validation failure or warning."""

    code: ValidationIssueCode
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    node_ids: tuple[str, ...] = ()
    operation_names: tuple[str, ...] = ()
    context: tuple[tuple[str, object], ...] = ()

    @classmethod
    def create(
            cls,
            code: ValidationIssueCode,
            message: str,
            *,
            severity: ValidationSeverity = ValidationSeverity.ERROR,
            node_ids: Iterable[object] = (),
            operation_names: Iterable[object] = (),
            context: Mapping[str, object] | None = None,
    ) -> "ValidationIssue":
        return cls(
            code=code,
            message=message,
            severity=severity,
            node_ids=_unique_strings(node_ids),
            operation_names=_unique_strings(operation_names),
            context=tuple(sorted((str(key), _freeze_value(value))
                                 for key, value in (context or {}).items())),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "node_ids": list(self.node_ids),
            "operation_names": list(self.operation_names),
            "context": {key: _json_value(value) for key, value in self.context},
        }


@dataclass(frozen=True)
class ValidationReport:
    """Complete deterministic result of validating one graph specification."""

    graph_id: str
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code.value for issue in self.issues
                     if issue.severity is ValidationSeverity.ERROR)

    def append(self, *issues: ValidationIssue) -> "ValidationReport":
        return ValidationReport(self.graph_id, self.issues + tuple(issues))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": GRAPH_VALIDATION_SCHEMA,
            "graph_id": self.graph_id,
            "is_valid": self.is_valid,
            "issues": [issue.to_record() for issue in self.issues],
        }
