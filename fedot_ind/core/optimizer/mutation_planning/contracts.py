"""Immutable edits and bounded outcomes for the structural mutation interpreter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fedot_ind.core.optimizer.graph_validation.contracts import (
    GraphSpec, OperationCapabilities, ValidationReport,
)


class MutationKind(str, Enum):
    REPLACE = "replace"
    ADD = "add"
    DROP = "drop"


class MutationRejectionCode(str, Enum):
    INVALID_GRAPH = "invalid_graph"
    MISSING_CONTEXT = "missing_context"
    UNSUPPORTED_EDIT = "unsupported_edit"
    STALE_PLAN = "stale_plan"
    BUDGET_EXCEEDED = "budget_exceeded"
    REPAIR_EXHAUSTED = "repair_exhausted"


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class MutationLimits:
    """Depth counts nodes, including the primary node and the final model."""

    max_nodes: int
    max_depth: int
    max_candidates: int = 64
    max_edits: int = 4
    max_arity: int | None = None

    def __post_init__(self) -> None:
        _integer("max_nodes", self.max_nodes, 1)
        _integer("max_depth", self.max_depth, 1)
        _integer("max_candidates", self.max_candidates)
        _integer("max_edits", self.max_edits)
        if self.max_arity is not None:
            _integer("max_arity", self.max_arity)


def _node_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node identifiers must be non-empty strings")


def _operation(value: OperationCapabilities) -> None:
    if not isinstance(value, OperationCapabilities):
        raise TypeError("operation must be an OperationCapabilities value")


@dataclass(frozen=True)
class RawInputSpec:
    """Confirmed homogeneous input, using the normalized GraphSpec data type.

    This describes the caller's existing data, not an operation's possible
    input types. It is inapplicable to explicit sources or multimodal graphs.
    """

    data_type: str
    has_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.data_type, str) or not self.data_type.strip():
            raise ValueError("data_type must be a non-empty string")
        if not isinstance(self.has_target, bool):
            raise TypeError("has_target must be boolean")

    def to_record(self) -> dict[str, object]:
        return {"data_type": self.data_type, "has_target": self.has_target}


@dataclass(frozen=True)
class ReplaceOperation:
    node_id: str
    operation: OperationCapabilities

    def __post_init__(self) -> None:
        _node_id(self.node_id)
        _operation(self.operation)


@dataclass(frozen=True)
class InsertTransform:
    """None reuses the child's implicit raw input; it never creates a data source."""

    parent_id: str | None
    child_id: str
    node_id: str
    operation: OperationCapabilities

    def __post_init__(self) -> None:
        for node_id in (self.child_id, self.node_id):
            _node_id(node_id)
        if self.parent_id is not None:
            _node_id(self.parent_id)
        _operation(self.operation)


@dataclass(frozen=True)
class DropTransform:
    node_id: str

    def __post_init__(self) -> None:
        _node_id(self.node_id)


@dataclass(frozen=True)
class AddBranch:
    """Reuse an existing parent of child, retaining its original incoming edge."""

    parent_id: str
    child_id: str
    node_id: str
    operation: OperationCapabilities

    def __post_init__(self) -> None:
        for node_id in (self.parent_id, self.child_id, self.node_id):
            _node_id(node_id)
        _operation(self.operation)


@dataclass(frozen=True)
class DropPrimaryTransform:
    """Remove a primary transform only against a confirmed RawInputSpec."""

    node_id: str

    def __post_init__(self) -> None:
        _node_id(self.node_id)


@dataclass(frozen=True)
class DeduplicateParents:
    """Repair only repeated references to the same node, not equal operations."""

    node_id: str

    def __post_init__(self) -> None:
        _node_id(self.node_id)


GraphEdit = (ReplaceOperation | InsertTransform | DropTransform | AddBranch
             | DropPrimaryTransform | DeduplicateParents)


def edit_record(edit: GraphEdit) -> dict[str, object]:
    if isinstance(edit, ReplaceOperation):
        return {"kind": "replace", "node_id": edit.node_id,
                "operation": edit.operation.to_record()}
    if isinstance(edit, InsertTransform):
        return {"kind": "add", "node_id": edit.node_id, "parent_id": edit.parent_id,
                "child_id": edit.child_id, "operation": edit.operation.to_record()}
    if isinstance(edit, DropTransform):
        return {"kind": "drop", "node_id": edit.node_id}
    if isinstance(edit, AddBranch):
        return {"kind": "add_branch", "node_id": edit.node_id, "parent_id": edit.parent_id,
                "child_id": edit.child_id, "operation": edit.operation.to_record()}
    if isinstance(edit, DropPrimaryTransform):
        return {"kind": "drop_primary", "node_id": edit.node_id}
    if isinstance(edit, DeduplicateParents):
        return {"kind": "deduplicate_parents", "node_id": edit.node_id}
    raise TypeError("Unsupported graph edit type")


@dataclass(frozen=True)
class MutationPlan:
    source_signature: str
    edits: tuple[GraphEdit, ...]
    raw_input: RawInputSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_signature, str) or not self.source_signature:
            raise ValueError("source_signature must not be empty")
        if not isinstance(self.edits, tuple) or not self.edits:
            raise ValueError("edits must be a non-empty tuple")
        if any(not isinstance(edit, (ReplaceOperation, InsertTransform,
                                     DropTransform, AddBranch, DropPrimaryTransform,
                                     DeduplicateParents)) for edit in self.edits):
            raise TypeError("Unsupported graph edit type")
        if self.raw_input is not None and not isinstance(self.raw_input, RawInputSpec):
            raise TypeError("raw_input must be a RawInputSpec or None")

    def to_record(self) -> dict[str, object]:
        record = {"schema_version": "industrial_mutation_plan@1",
                  "source_signature": self.source_signature,
                  "edits": [edit_record(edit) for edit in self.edits]}
        if self.raw_input is not None:
            record["raw_input"] = self.raw_input.to_record()
        return record


@dataclass(frozen=True)
class MutationRejection:
    code: MutationRejectionCode
    message: str
    report: ValidationReport | None = None
    examined: int = 0


@dataclass(frozen=True)
class MutationCandidates:
    plans: tuple[MutationPlan, ...]
    examined: int
    truncated: bool
    rejection: MutationRejection | None = None


@dataclass(frozen=True)
class AppliedMutation:
    graph: GraphSpec
    report: ValidationReport


@dataclass(frozen=True)
class RepairedGraph:
    graph: GraphSpec
    edits: tuple[GraphEdit, ...]
    examined: int
    report: ValidationReport
