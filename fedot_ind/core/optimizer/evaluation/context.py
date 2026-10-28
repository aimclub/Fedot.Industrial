"""Per-call evaluation evidence, scoped to the owning worker."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class EvaluationBudget:
    preparation_seconds: float
    remaining_search_seconds: float | None
    execution_budget_seconds: float | None
    expired_before_execution: bool


@dataclass(frozen=True)
class EvaluationContext:
    evaluation_id: str
    candidate_id: str
    configuration_id: str
    seed: int | None
    configuration_json: str | None = None
    budget: EvaluationBudget | None = None

    def to_record(self):
        return {
            'evaluation_id': self.evaluation_id, 'candidate_id': self.candidate_id,
            'configuration_id': self.configuration_id, 'seed': self.seed,
            'rng_scope': 'python_numpy',
            'configuration': None if self.configuration_json is None else json.loads(self.configuration_json),
            'budget': None if self.budget is None else asdict(self.budget),
        }


_CURRENT: ContextVar[EvaluationContext | None] = ContextVar('industrial_evaluation_context', default=None)


def current_evaluation_context() -> EvaluationContext | None:
    return _CURRENT.get()


@contextmanager
def evaluation_context_scope(context: EvaluationContext | None):
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)
