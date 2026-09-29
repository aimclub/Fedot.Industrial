"""Typed observability records for Industrial evolutionary optimisation."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Mapping, Sequence


EVOLUTION_DIAGNOSTICS_SCHEMA = "industrial_evolution_diagnostics@1"


class EvolutionEventType(str, Enum):
    """Kinds of observations emitted by the evolutionary runtime."""

    MUTATION_ATTEMPT = "mutation_attempt"
    CANDIDATE_DECISION = "candidate_decision"
    EVALUATION = "evaluation"
    GENERATION = "generation"


class CandidateStatus(str, Enum):
    """Result of checking one generated offspring candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class CandidateRejectionReason(str, Enum):
    """Stable reasons why a generated candidate was not admitted."""

    MUTATION_DID_NOT_RETURN_INDIVIDUAL = "mutation_did_not_return_individual"
    VERIFIER_REJECTED = "verifier_rejected"
    VERIFIER_FAILED = "verifier_failed"
    DUPLICATE_GRAPH = "duplicate_graph"


class EvaluationStatus(str, Enum):
    """Result of evaluating or reusing an individual."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REUSED = "reused"


@dataclass(frozen=True)
class EvolutionObservation:
    """One ordered, serializable observation from an optimisation run."""

    sequence: int
    event_type: EvolutionEventType
    elapsed_seconds: float
    generation: int | None = None
    individual_id: str | None = None
    graph_id: str | None = None
    status: str | None = None
    reason: str | None = None
    duration_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    population_size: int | None = None
    unique_graph_count: int | None = None
    best_fitness: float | None = None
    label: str | None = None

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        values = {
            "schema_version": EVOLUTION_DIAGNOSTICS_SCHEMA,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "elapsed_seconds": self.elapsed_seconds,
            "generation": self.generation,
            "individual_id": self.individual_id,
            "graph_id": self.graph_id,
            "status": self.status,
            "reason": self.reason,
            "duration_seconds": self.duration_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "population_size": self.population_size,
            "unique_graph_count": self.unique_graph_count,
            "best_fitness": self.best_fitness,
            "label": self.label,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class EvolutionSummary:
    """Aggregated diagnostics derived from an ordered event stream."""

    mutation_attempts: int
    candidates_accepted: int
    candidates_rejected: int
    valid_offspring_ratio: float
    rejection_counts: Mapping[str, int]
    evaluations_succeeded: int
    evaluations_failed: int
    evaluations_reused: int
    evaluation_success_ratio: float
    evaluation_failure_counts: Mapping[str, int]
    evaluation_duration_seconds: float
    mean_evaluation_seconds: float
    generations_observed: int
    population_observations: int
    unique_graph_observations: int
    uniqueness_ratio: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary."""
        return {
            "schema_version": EVOLUTION_DIAGNOSTICS_SCHEMA,
            "mutation_attempts": self.mutation_attempts,
            "candidates_accepted": self.candidates_accepted,
            "candidates_rejected": self.candidates_rejected,
            "valid_offspring_ratio": self.valid_offspring_ratio,
            "rejection_counts": dict(self.rejection_counts),
            "evaluations_succeeded": self.evaluations_succeeded,
            "evaluations_failed": self.evaluations_failed,
            "evaluations_reused": self.evaluations_reused,
            "evaluation_success_ratio": self.evaluation_success_ratio,
            "evaluation_failure_counts": dict(self.evaluation_failure_counts),
            "evaluation_duration_seconds": self.evaluation_duration_seconds,
            "mean_evaluation_seconds": self.mean_evaluation_seconds,
            "generations_observed": self.generations_observed,
            "population_observations": self.population_observations,
            "unique_graph_observations": self.unique_graph_observations,
            "uniqueness_ratio": self.uniqueness_ratio,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class EvolutionDiagnosticsSnapshot:
    """Immutable snapshot of observations and their derived summary."""

    observations: tuple[EvolutionObservation, ...]
    summary: EvolutionSummary

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": EVOLUTION_DIAGNOSTICS_SCHEMA,
            "summary": self.summary.to_record(),
            "events": [observation.to_record() for observation in self.observations],
        }


@dataclass(frozen=True)
class EvolutionArtifactPaths:
    """Paths produced by :func:`write_evolution_diagnostics`."""

    events_jsonl: Path
    summary_json: Path
    summary_markdown: Path


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_evolution_summary(
        observations: Iterable[EvolutionObservation],
) -> EvolutionSummary:
    """Aggregate diagnostics without depending on optimiser runtime state."""
    events = tuple(observations)
    mutation_attempts = sum(
        event.event_type is EvolutionEventType.MUTATION_ATTEMPT for event in events
    )
    candidate_events = tuple(
        event for event in events
        if event.event_type is EvolutionEventType.CANDIDATE_DECISION
    )
    candidates_accepted = sum(
        event.status == CandidateStatus.ACCEPTED.value for event in candidate_events
    )
    candidates_rejected = sum(
        event.status == CandidateStatus.REJECTED.value for event in candidate_events
    )
    rejection_counts = Counter(
        event.reason for event in candidate_events
        if event.status == CandidateStatus.REJECTED.value and event.reason
    )

    evaluation_events = tuple(
        event for event in events if event.event_type is EvolutionEventType.EVALUATION
    )
    evaluations_succeeded = sum(
        event.status == EvaluationStatus.SUCCEEDED.value for event in evaluation_events
    )
    evaluations_failed = sum(
        event.status == EvaluationStatus.FAILED.value for event in evaluation_events
    )
    evaluations_reused = sum(
        event.status == EvaluationStatus.REUSED.value for event in evaluation_events
    )
    evaluation_failure_counts = Counter(
        event.error_type or "unknown_error" for event in evaluation_events
        if event.status == EvaluationStatus.FAILED.value
    )
    evaluation_durations = tuple(
        event.duration_seconds for event in evaluation_events
        if event.duration_seconds is not None
    )

    generation_events = tuple(
        event for event in events if event.event_type is EvolutionEventType.GENERATION
    )
    population_observations = sum(
        event.population_size or 0 for event in generation_events)
    unique_graph_observations = sum(
        event.unique_graph_count or 0 for event in generation_events)
    terminal_elapsed = max(
        (event.elapsed_seconds for event in events), default=0.0)
    evaluated = evaluations_succeeded + evaluations_failed

    return EvolutionSummary(
        mutation_attempts=mutation_attempts,
        candidates_accepted=candidates_accepted,
        candidates_rejected=candidates_rejected,
        valid_offspring_ratio=_safe_ratio(
            candidates_accepted, mutation_attempts),
        rejection_counts=dict(sorted(rejection_counts.items())),
        evaluations_succeeded=evaluations_succeeded,
        evaluations_failed=evaluations_failed,
        evaluations_reused=evaluations_reused,
        evaluation_success_ratio=_safe_ratio(evaluations_succeeded, evaluated),
        evaluation_failure_counts=dict(
            sorted(evaluation_failure_counts.items())),
        evaluation_duration_seconds=float(sum(evaluation_durations)),
        mean_evaluation_seconds=float(
            fmean(evaluation_durations)) if evaluation_durations else 0.0,
        generations_observed=len(generation_events),
        population_observations=population_observations,
        unique_graph_observations=unique_graph_observations,
        uniqueness_ratio=_safe_ratio(
            unique_graph_observations, population_observations),
        elapsed_seconds=float(terminal_elapsed),
    )


class EvolutionDiagnosticsRecorder:
    """Thread-safe effect shell that records ordered optimisation events."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started_at = float(clock())
        self._observations: list[EvolutionObservation] = []
        self._lock = threading.Lock()

    def _record(self, event_type: EvolutionEventType, **values) -> EvolutionObservation:
        with self._lock:
            observation = EvolutionObservation(
                sequence=len(self._observations),
                event_type=event_type,
                elapsed_seconds=max(0.0, float(
                    self._clock()) - self._started_at),
                **values,
            )
            self._observations.append(observation)
        return observation

    def record_mutation_attempt(
            self,
            *,
            generation: int | None,
            individual_id: str | None,
            graph_id: str | None,
    ) -> EvolutionObservation:
        return self._record(
            EvolutionEventType.MUTATION_ATTEMPT,
            generation=generation,
            individual_id=individual_id,
            graph_id=graph_id,
        )

    def record_candidate(
            self,
            *,
            generation: int | None,
            individual_id: str | None,
            graph_id: str | None,
            status: CandidateStatus,
            reason: CandidateRejectionReason | None = None,
            error: BaseException | None = None,
    ) -> EvolutionObservation:
        return self._record(
            EvolutionEventType.CANDIDATE_DECISION,
            generation=generation,
            individual_id=individual_id,
            graph_id=graph_id,
            status=status.value,
            reason=reason.value if reason is not None else None,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    def record_evaluation(
            self,
            *,
            individual_id: str | None,
            graph_id: str | None,
            status: EvaluationStatus,
            duration_seconds: float | None = None,
            error_type: str | None = None,
            error_message: str | None = None,
    ) -> EvolutionObservation:
        duration = None
        if duration_seconds is not None and math.isfinite(float(duration_seconds)):
            duration = max(0.0, float(duration_seconds))
        return self._record(
            EvolutionEventType.EVALUATION,
            individual_id=individual_id,
            graph_id=graph_id,
            status=status.value,
            duration_seconds=duration,
            error_type=error_type,
            error_message=error_message,
        )

    def record_generation(
            self,
            *,
            generation: int | None,
            graph_ids: Sequence[str],
            best_fitness: float | None,
            label: str | None,
    ) -> EvolutionObservation:
        normalized_ids = tuple(str(graph_id) for graph_id in graph_ids)
        return self._record(
            EvolutionEventType.GENERATION,
            generation=generation,
            population_size=len(normalized_ids),
            unique_graph_count=len(set(normalized_ids)),
            best_fitness=best_fitness,
            label=label,
        )

    def snapshot(self) -> EvolutionDiagnosticsSnapshot:
        with self._lock:
            observations = tuple(self._observations)
        return EvolutionDiagnosticsSnapshot(
            observations=observations,
            summary=build_evolution_summary(observations),
        )


def graph_identity(graph: object) -> str:
    """Return a stable diagnostic identity without retaining the graph."""
    descriptive_id = getattr(graph, "descriptive_id", None)
    if descriptive_id is not None:
        return str(descriptive_id)
    uid = getattr(graph, "uid", None)
    if uid is not None:
        return str(uid)
    return type(graph).__name__


def fitness_scalar(fitness: object) -> float | None:
    """Extract the first finite fitness value for diagnostic summaries."""
    values = getattr(fitness, "values", None)
    if values is None:
        values = (getattr(fitness, "value", None),)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = (values,)
    for value in values:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            return scalar
    return None


def _summary_markdown(summary: EvolutionSummary) -> str:
    rejection_rows = "\n".join(
        f"| `{reason}` | {count} |" for reason, count in summary.rejection_counts.items()
    ) or "| none | 0 |"
    failure_rows = "\n".join(
        f"| `{reason}` | {count} |" for reason, count in summary.evaluation_failure_counts.items()
    ) or "| none | 0 |"
    return "\n".join((
        "# Evolution diagnostics",
        "",
        f"Schema: `{EVOLUTION_DIAGNOSTICS_SCHEMA}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mutation attempts | {summary.mutation_attempts} |",
        f"| Accepted candidates | {summary.candidates_accepted} |",
        f"| Rejected candidates | {summary.candidates_rejected} |",
        f"| Valid offspring ratio | {summary.valid_offspring_ratio:.6f} |",
        f"| Successful evaluations | {summary.evaluations_succeeded} |",
        f"| Failed evaluations | {summary.evaluations_failed} |",
        f"| Reused evaluations | {summary.evaluations_reused} |",
        f"| Evaluation success ratio | {summary.evaluation_success_ratio:.6f} |",
        f"| Generations observed | {summary.generations_observed} |",
        f"| Uniqueness ratio | {summary.uniqueness_ratio:.6f} |",
        f"| Evaluation time, seconds | {summary.evaluation_duration_seconds:.6f} |",
        f"| Run elapsed time, seconds | {summary.elapsed_seconds:.6f} |",
        "",
        "## Candidate rejections",
        "",
        "| Reason | Count |",
        "|---|---:|",
        rejection_rows,
        "",
        "## Evaluation failures",
        "",
        "| Error type | Count |",
        "|---|---:|",
        failure_rows,
        "",
    ))


def write_evolution_diagnostics(
        snapshot: EvolutionDiagnosticsSnapshot,
        output_dir: str | Path,
) -> EvolutionArtifactPaths:
    """Persist a snapshot as inspectable JSONL, JSON and Markdown artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    events_path = target / "evolution_events.jsonl"
    summary_path = target / "evolution_summary.json"
    markdown_path = target / "evolution_summary.md"

    with events_path.open("w", encoding="utf-8", newline="\n") as stream:
        for observation in snapshot.observations:
            stream.write(json.dumps(observation.to_record(),
                         ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    summary_path.write_text(
        json.dumps(snapshot.summary.to_record(), ensure_ascii=False,
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_summary_markdown(
        snapshot.summary), encoding="utf-8")
    return EvolutionArtifactPaths(
        events_jsonl=events_path,
        summary_json=summary_path,
        summary_markdown=markdown_path,
    )
