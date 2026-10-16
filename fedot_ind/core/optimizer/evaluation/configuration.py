"""Pure transport boundary for the opt-in local executor."""

from dataclasses import dataclass
from enum import Enum

from .contracts import positive_seconds


class EvaluationBackend(str, Enum):
    LEGACY = "legacy"
    SEQUENTIAL = "sequential"
    PROCESS = "process"


@dataclass(frozen=True)
class EvaluationExecutionConfig:
    backend: EvaluationBackend = EvaluationBackend.LEGACY
    candidate_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.backend, EvaluationBackend):
            raise ValueError("backend must be an EvaluationBackend")
        positive_seconds(self.candidate_timeout_seconds, "candidate_timeout_seconds")
        positive_seconds(self.total_timeout_seconds, "total_timeout_seconds")
        if self.backend is EvaluationBackend.LEGACY and (
            self.candidate_timeout_seconds is not None or self.total_timeout_seconds is not None
        ):
            raise ValueError("legacy backend cannot enforce local executor deadlines")

    def to_record(self):
        return {"backend": self.backend.value, "candidate_timeout_seconds": self.candidate_timeout_seconds,
                "total_timeout_seconds": self.total_timeout_seconds}
