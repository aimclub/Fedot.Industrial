"""Immutable execution decisions; no live graphs or resources in a plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class LocalBackend(str, Enum):
    SEQUENTIAL = "sequential"
    PROCESS = "process"


class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    WORKER_FAILED = "worker_failed"


def positive_seconds(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (float, int))
        or not math.isfinite(value) or value <= 0
    ):
        raise ValueError(f"{name} must be finite positive seconds or null")


@dataclass(frozen=True)
class EvaluationTask:
    candidate_id: str
    seed: int | None = None

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be a uint32 or null")


@dataclass(frozen=True)
class EvaluationPlan:
    tasks: tuple[EvaluationTask, ...]
    backend: LocalBackend = LocalBackend.SEQUENTIAL
    max_workers: int = 1
    candidate_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.tasks, tuple) or not all(isinstance(t, EvaluationTask) for t in self.tasks):
            raise ValueError("tasks must be a tuple of EvaluationTask")
        if len({task.candidate_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("candidate ids must be unique within a plan")
        if not isinstance(self.backend, LocalBackend):
            raise ValueError("backend must be a LocalBackend")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if self.backend is LocalBackend.SEQUENTIAL and self.max_workers != 1:
            raise ValueError("sequential execution requires max_workers=1")
        positive_seconds(self.candidate_timeout_seconds, "candidate_timeout_seconds")
        positive_seconds(self.total_timeout_seconds, "total_timeout_seconds")


@dataclass(frozen=True)
class ExecutionError:
    error_type: str
    message: str


@dataclass(frozen=True)
class ExecutionResult:
    candidate_id: str
    status: ExecutionStatus
    duration_seconds: float
    value: object = None
    error: ExecutionError | None = None

    def __post_init__(self):
        if not isinstance(self.status, ExecutionStatus):
            raise ValueError("status must be an ExecutionStatus")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if (self.status is ExecutionStatus.SUCCEEDED) != (self.error is None):
            raise ValueError("only successful execution may omit error")
        if self.status is not ExecutionStatus.SUCCEEDED and self.value is not None:
            raise ValueError("failed execution cannot carry a successful value")
