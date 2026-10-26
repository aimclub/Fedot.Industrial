"""Pure configuration and execution plans for the opt-in Dask executor."""

from dataclasses import dataclass
from typing import Mapping

from .contracts import EvaluationTask, positive_seconds


@dataclass(frozen=True)
class DaskConfigViolation:
    code: str
    path: str
    message: str
    value: object = None


class DaskConfigError(ValueError):
    def __init__(self, violations: tuple[DaskConfigViolation, ...]):
        self.violations = violations
        super().__init__("; ".join(f"{item.path}: {item.message}" for item in violations))


def _validate_options(options: Mapping) -> tuple[DaskConfigViolation, ...]:
    violations = []
    for name, value in options.items():
        message = None
        if name == "retries":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
                message = "must be an integer between 0 and 3"
        elif name == "memory_limit":
            if not (isinstance(value, str) and value == "auto") and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                message = "must be 'auto' or positive integer bytes"
        else:
            try:
                positive_seconds(value, name)
            except (ValueError, OverflowError):
                message = "must be finite positive seconds"
            if value is None:
                message = "must be finite positive seconds"
        if message is not None:
            violations.append(DaskConfigViolation("invalid_value", name, message, value))
    return tuple(violations)


@dataclass(frozen=True)
class DaskExecutionConfig:
    retries: int = 0
    startup_timeout_seconds: float = 60
    restart_timeout_seconds: float = 30
    poll_interval_seconds: float = .05
    memory_limit: str | int = "auto"

    def __post_init__(self):
        violations = _validate_options(self.to_record())
        if violations:
            raise DaskConfigError(violations)

    def to_record(self) -> dict[str, object]:
        return {
            "retries": self.retries,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "restart_timeout_seconds": self.restart_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "memory_limit": self.memory_limit,
        }


def parse_dask_config(payload: Mapping[str, object]) -> DaskExecutionConfig:
    """Reject unknown keys and report all independent option violations."""
    if not isinstance(payload, Mapping):
        raise DaskConfigError((DaskConfigViolation("invalid_section", "", "must be an object", payload),))
    allowed = DaskExecutionConfig().to_record()
    violations = [
        DaskConfigViolation("unknown_field", str(name), "is not supported", payload[name])
        for name in sorted(set(payload) - set(allowed), key=str)
    ]
    options = {name: payload[name] for name in allowed if name in payload}
    violations.extend(_validate_options(options))
    if violations:
        raise DaskConfigError(tuple(violations))
    return DaskExecutionConfig(**options)


@dataclass(frozen=True)
class DaskEvaluationPlan:
    tasks: tuple[EvaluationTask, ...]
    max_workers: int = 1
    candidate_timeout_seconds: float | None = None
    total_timeout_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.tasks, tuple) or not all(isinstance(task, EvaluationTask) for task in self.tasks):
            raise ValueError("tasks must be a tuple of EvaluationTask")
        if len({task.candidate_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("candidate ids must be unique within a plan")
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        positive_seconds(self.candidate_timeout_seconds, "candidate_timeout_seconds")
        positive_seconds(self.total_timeout_seconds, "total_timeout_seconds")
