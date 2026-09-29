"""Typed states, failures and outcomes for Industrial evolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeAlias


EVALUATION_OUTCOME_SCHEMA = "industrial_evaluation_outcome@1"


class EvolutionPhase(str, Enum):
    """Closed lifecycle states of one optimisation run."""

    CREATED = "created"
    INITIALISING = "initialising"
    EVALUATING_INITIAL = "evaluating_initial"
    EVOLVING = "evolving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class EvolutionState:
    """Immutable lifecycle state with monotonic generation progress."""

    phase: EvolutionPhase
    generation: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise ValueError("generation must be an integer")
        if self.generation < 0:
            raise ValueError("generation must not be negative")

    @classmethod
    def created(cls) -> "EvolutionState":
        return cls(phase=EvolutionPhase.CREATED)


class EvolutionTransitionError(RuntimeError):
    """Raised when lifecycle movement violates the state contract."""

    def __init__(self, current: EvolutionState, target: EvolutionPhase, message: str):
        self.current = current
        self.target = target
        super().__init__(
            f"Cannot transition evolution from {current.phase.value} to {target.value}: {message}"
        )


_ALLOWED_TRANSITIONS = {
    EvolutionPhase.CREATED: frozenset({EvolutionPhase.INITIALISING, EvolutionPhase.FAILED}),
    EvolutionPhase.INITIALISING: frozenset({EvolutionPhase.EVALUATING_INITIAL, EvolutionPhase.FAILED}),
    EvolutionPhase.EVALUATING_INITIAL: frozenset({
        EvolutionPhase.EVOLVING,
        EvolutionPhase.COMPLETED,
        EvolutionPhase.FAILED,
    }),
    EvolutionPhase.EVOLVING: frozenset({
        EvolutionPhase.EVOLVING,
        EvolutionPhase.COMPLETED,
        EvolutionPhase.FAILED,
    }),
    EvolutionPhase.COMPLETED: frozenset(),
    EvolutionPhase.FAILED: frozenset(),
}


def transition_evolution_state(
        state: EvolutionState,
        target: EvolutionPhase,
        *,
        generation: int | None = None,
) -> EvolutionState:
    """Apply one validated, deterministic lifecycle transition."""
    next_generation = state.generation if generation is None else generation
    if target not in _ALLOWED_TRANSITIONS[state.phase]:
        raise EvolutionTransitionError(state, target, "transition is not allowed")
    if next_generation < state.generation:
        raise EvolutionTransitionError(state, target, "generation must be monotonic")
    return EvolutionState(phase=target, generation=next_generation)


class CandidateFailureCode(str, Enum):
    """Expected reasons why an offspring cannot enter a population."""

    MUTATION_DID_NOT_RETURN_INDIVIDUAL = "mutation_did_not_return_individual"
    VERIFIER_REJECTED = "verifier_rejected"
    VERIFIER_FAILED = "verifier_failed"
    DUPLICATE_GRAPH = "duplicate_graph"


@dataclass(frozen=True)
class CandidateFailure:
    """Structured rejection with stable identifiers and a human message."""

    code: CandidateFailureCode
    message: str
    individual_id: str | None = None
    graph_id: str | None = None
    validation_issue_codes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        record = {
            "code": self.code.value,
            "message": self.message,
            "individual_id": self.individual_id,
            "graph_id": self.graph_id,
            "validation_issue_codes": list(self.validation_issue_codes),
        }
        return {key: value for key, value in record.items()
                if value is not None and value != []}


def classify_candidate_failure(
        *,
        is_individual: bool,
        is_valid_graph: bool | None,
        is_duplicate_graph: bool | None,
        individual_id: str | None = None,
        graph_id: str | None = None,
        validation_issue_codes: tuple[str, ...] = (),
) -> CandidateFailure | None:
    """Classify expected candidate rejection without mutating optimiser state."""
    if not is_individual:
        return CandidateFailure(
            code=CandidateFailureCode.MUTATION_DID_NOT_RETURN_INDIVIDUAL,
            message="Mutation did not return an Individual",
        )
    if is_valid_graph is False:
        return CandidateFailure(
            code=CandidateFailureCode.VERIFIER_REJECTED,
            message="Graph verifier rejected the candidate",
            individual_id=individual_id,
            graph_id=graph_id,
            validation_issue_codes=validation_issue_codes,
        )
    if is_duplicate_graph is True:
        return CandidateFailure(
            code=CandidateFailureCode.DUPLICATE_GRAPH,
            message="Candidate duplicates a graph already present in the population",
            individual_id=individual_id,
            graph_id=graph_id,
        )
    return None


class EvaluationFailureCode(str, Enum):
    """Expected categories of candidate evaluation failure."""

    OBJECTIVE_EXCEPTION = "objective_exception"
    INVALID_FITNESS = "invalid_fitness"


@dataclass(frozen=True)
class EvaluationFailure:
    """Structured failure retained across worker and coordinator boundaries."""

    code: EvaluationFailureCode
    error_type: str
    message: str

    def to_record(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class EvaluationSucceeded:
    """Successful objective evaluation."""

    duration_seconds: float
    evaluated_at: str


@dataclass(frozen=True)
class EvaluationFailed:
    """Failed objective evaluation represented as data."""

    duration_seconds: float
    evaluated_at: str
    failure: EvaluationFailure


@dataclass(frozen=True)
class EvaluationReused:
    """An existing valid fitness reused without objective execution."""

    duration_seconds: float = 0.0


EvaluationOutcome: TypeAlias = EvaluationSucceeded | EvaluationFailed | EvaluationReused


class EvaluationOutcomeDecodeError(ValueError):
    """Raised when worker metadata violates the evaluation outcome schema."""


def evaluation_outcome_to_metadata(outcome: EvaluationOutcome) -> dict[str, object]:
    """Serialize one outcome into GraphEvalResult-compatible metadata."""
    if isinstance(outcome, EvaluationSucceeded):
        return {
            "evaluation_schema": EVALUATION_OUTCOME_SCHEMA,
            "evaluation_status": "succeeded",
            "computation_time_in_seconds": outcome.duration_seconds,
            "evaluation_time_iso": outcome.evaluated_at,
        }
    if isinstance(outcome, EvaluationReused):
        return {
            "evaluation_schema": EVALUATION_OUTCOME_SCHEMA,
            "evaluation_status": "reused",
            "computation_time_in_seconds": outcome.duration_seconds,
        }
    return {
        "evaluation_schema": EVALUATION_OUTCOME_SCHEMA,
        "evaluation_status": "failed",
        "computation_time_in_seconds": outcome.duration_seconds,
        "evaluation_time_iso": outcome.evaluated_at,
        "evaluation_failure": outcome.failure.to_record(),
        # Legacy keys remain until every downstream consumer reads the typed payload.
        "evaluation_error": outcome.failure.message,
        "evaluation_error_type": outcome.failure.error_type,
    }


def evaluation_outcome_from_metadata(metadata: Mapping[str, object]) -> EvaluationOutcome:
    """Decode canonical metadata and tolerate the pre-EVO-02 failure keys."""
    schema = metadata.get("evaluation_schema")
    if schema not in (None, EVALUATION_OUTCOME_SCHEMA):
        raise EvaluationOutcomeDecodeError(f"Unsupported evaluation schema: {schema}")
    status = metadata.get("evaluation_status")
    if status not in (None, "succeeded", "failed", "reused"):
        raise EvaluationOutcomeDecodeError(f"Unsupported evaluation status: {status}")
    try:
        duration = float(metadata.get("computation_time_in_seconds", 0.0))
    except (TypeError, ValueError) as error:
        raise EvaluationOutcomeDecodeError("Evaluation duration must be numeric") from error
    evaluated_at = str(metadata.get("evaluation_time_iso", ""))
    if status == "reused":
        return EvaluationReused(duration_seconds=duration)
    failure_payload = metadata.get("evaluation_failure")
    if isinstance(failure_payload, Mapping):
        try:
            code = EvaluationFailureCode(failure_payload.get("code"))
        except (TypeError, ValueError) as error:
            raise EvaluationOutcomeDecodeError("Unsupported evaluation failure code") from error
        return EvaluationFailed(
            duration_seconds=duration,
            evaluated_at=evaluated_at,
            failure=EvaluationFailure(
                code=code,
                error_type=str(failure_payload.get("error_type", "UnknownEvaluationError")),
                message=str(failure_payload.get("message", "Evaluation failed")),
            ),
        )
    legacy_error = metadata.get("evaluation_error")
    if status == "failed" or legacy_error is not None:
        error_type = str(metadata.get("evaluation_error_type", "UnknownEvaluationError"))
        return EvaluationFailed(
            duration_seconds=duration,
            evaluated_at=evaluated_at,
            failure=EvaluationFailure(
                code=(
                    EvaluationFailureCode.INVALID_FITNESS
                    if error_type == "InvalidFitness"
                    else EvaluationFailureCode.OBJECTIVE_EXCEPTION
                ),
                error_type=error_type,
                message=str(legacy_error or "Evaluation failed"),
            ),
        )
    return EvaluationSucceeded(duration_seconds=duration, evaluated_at=evaluated_at)


class EvolutionRuntimeErrorCode(str, Enum):
    """Stable runtime failure categories exposed by Industrial evolution."""

    INITIAL_POPULATION_EMPTY = "initial_population_empty"


class EvolutionRuntimeError(RuntimeError):
    """Base exception carrying a stable machine-readable error code."""

    def __init__(self, code: EvolutionRuntimeErrorCode, message: str):
        self.code = code
        super().__init__(message)


class IndustrialPopulationError(EvolutionRuntimeError):
    """Raised when no valid individual can seed Industrial optimisation."""

    def __init__(self, message: str, *, timed_out: bool = False):
        self.timed_out = timed_out
        super().__init__(EvolutionRuntimeErrorCode.INITIAL_POPULATION_EMPTY, message)
