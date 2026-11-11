"""Wall-clock phases of a completed coordinator batch, not worker CPU sums."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BatchTiming:
    preparation_seconds: float
    execution_seconds: float
    application_seconds: float
    candidates: int

    def to_record(self):
        return asdict(self)
