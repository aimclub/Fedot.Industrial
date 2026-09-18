from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark.industrial.core import to_plain_data


@dataclass(frozen=True)
class ToolArtifact:
    kind: str
    path: str
    format: str = ''


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRequest:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, Any] | None = None) -> "ToolRequest":
        request_payload = dict(payload or {})
        execute = bool(request_payload.pop("execute", False))
        dry_run = bool(request_payload.pop("dry_run", not execute))
        return cls(name=name, payload=request_payload, dry_run=dry_run)


@dataclass(frozen=True)
class ToolResponse:
    name: str
    status: str
    dry_run: bool
    message: str = ''
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[ToolArtifact, ...] = ()
    error: ToolError | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)


def artifact_from_path(path: str | Path, *, kind: str, format_: str = '') -> ToolArtifact:
    artifact_path = Path(path)
    return ToolArtifact(kind=kind, path=str(artifact_path), format=format_ or artifact_path.suffix.lstrip('.'))


__all__ = [
    "ToolArtifact",
    "ToolError",
    "ToolRequest",
    "ToolResponse",
    "artifact_from_path",
]
