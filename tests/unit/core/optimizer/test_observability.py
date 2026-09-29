import json
from types import SimpleNamespace

import pytest

from fedot_ind.core.optimizer.observability import (
    CandidateRejectionReason,
    CandidateStatus,
    EvaluationStatus,
    EvolutionDiagnosticsRecorder,
    EvolutionEventType,
    build_evolution_summary,
    fitness_scalar,
    graph_identity,
    write_evolution_diagnostics,
)


class _Clock:
    def __init__(self, *values: float):
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _populated_recorder() -> EvolutionDiagnosticsRecorder:
    recorder = EvolutionDiagnosticsRecorder(clock=_Clock(
        10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7))
    recorder.record_mutation_attempt(
        generation=0, individual_id="parent", graph_id="g0")
    recorder.record_candidate(
        generation=0,
        individual_id="candidate-1",
        graph_id="g1",
        status=CandidateStatus.ACCEPTED,
    )
    recorder.record_mutation_attempt(
        generation=0, individual_id="parent", graph_id="g0")
    recorder.record_candidate(
        generation=0,
        individual_id="candidate-2",
        graph_id="g0",
        status=CandidateStatus.REJECTED,
        reason=CandidateRejectionReason.DUPLICATE_GRAPH,
    )
    recorder.record_evaluation(
        individual_id="candidate-1",
        graph_id="g1",
        status=EvaluationStatus.SUCCEEDED,
        duration_seconds=0.25,
    )
    recorder.record_evaluation(
        individual_id="candidate-2",
        graph_id="g2",
        status=EvaluationStatus.FAILED,
        duration_seconds=0.75,
        error_type="ValueError",
        error_message="invalid pipeline",
    )
    recorder.record_generation(
        generation=0,
        graph_ids=("g0", "g1", "g1"),
        best_fitness=0.1,
        label="initial_assumptions",
    )
    return recorder


def test_recorder_builds_stable_ordered_summary():
    snapshot = _populated_recorder().snapshot()

    assert [event.sequence for event in snapshot.observations] == list(
        range(7))
    assert snapshot.observations[0].event_type is EvolutionEventType.MUTATION_ATTEMPT
    assert snapshot.observations[-1].elapsed_seconds == pytest.approx(0.7)
    assert snapshot.summary.mutation_attempts == 2
    assert snapshot.summary.candidates_accepted == 1
    assert snapshot.summary.candidates_rejected == 1
    assert snapshot.summary.valid_offspring_ratio == pytest.approx(0.5)
    assert snapshot.summary.rejection_counts == {"duplicate_graph": 1}
    assert snapshot.summary.evaluations_succeeded == 1
    assert snapshot.summary.evaluations_failed == 1
    assert snapshot.summary.evaluation_success_ratio == pytest.approx(0.5)
    assert snapshot.summary.evaluation_failure_counts == {"ValueError": 1}
    assert snapshot.summary.evaluation_duration_seconds == pytest.approx(1.0)
    assert snapshot.summary.mean_evaluation_seconds == pytest.approx(0.5)
    assert snapshot.summary.generations_observed == 1
    assert snapshot.summary.uniqueness_ratio == pytest.approx(2 / 3)


def test_summary_is_total_for_empty_event_stream():
    summary = build_evolution_summary(())

    assert summary.mutation_attempts == 0
    assert summary.valid_offspring_ratio == 0.0
    assert summary.evaluation_success_ratio == 0.0
    assert summary.uniqueness_ratio == 0.0
    assert summary.elapsed_seconds == 0.0


def test_summary_counts_reused_evaluations_without_changing_success_ratio():
    recorder = EvolutionDiagnosticsRecorder(clock=_Clock(0.0, 0.1, 0.2, 0.3))
    recorder.record_evaluation(
        individual_id="one",
        graph_id="g1",
        status=EvaluationStatus.SUCCEEDED,
    )
    recorder.record_evaluation(
        individual_id="two",
        graph_id="g2",
        status=EvaluationStatus.REUSED,
    )
    recorder.record_evaluation(
        individual_id="three",
        graph_id="g3",
        status=EvaluationStatus.REUSED,
    )

    summary = recorder.snapshot().summary

    assert summary.evaluations_succeeded == 1
    assert summary.evaluations_reused == 2
    assert summary.evaluation_success_ratio == 1.0


def test_invalid_or_negative_evaluation_duration_is_not_aggregated():
    recorder = EvolutionDiagnosticsRecorder(clock=_Clock(0.0, 0.1, 0.2))
    recorder.record_evaluation(
        individual_id="one",
        graph_id="g1",
        status=EvaluationStatus.SUCCEEDED,
        duration_seconds=float("inf"),
    )
    recorder.record_evaluation(
        individual_id="two",
        graph_id="g2",
        status=EvaluationStatus.SUCCEEDED,
        duration_seconds=-1.0,
    )

    observations = recorder.snapshot().observations

    assert observations[0].duration_seconds is None
    assert observations[1].duration_seconds == 0.0


def test_snapshot_export_keeps_source_data_next_to_summary(tmp_path):
    snapshot = _populated_recorder().snapshot()

    paths = write_evolution_diagnostics(snapshot, tmp_path)

    event_records = [json.loads(line) for line in paths.events_jsonl.read_text(
        encoding="utf-8").splitlines()]
    summary_record = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    markdown = paths.summary_markdown.read_text(encoding="utf-8")
    assert len(event_records) == len(snapshot.observations)
    assert event_records[0]["schema_version"] == "industrial_evolution_diagnostics@1"
    assert summary_record["mutation_attempts"] == 2
    assert "Valid offspring ratio" in markdown
    assert "`duplicate_graph`" in markdown


@pytest.mark.parametrize(
    "graph, expected",
    [
        (SimpleNamespace(descriptive_id="pipeline/a"), "pipeline/a"),
        (SimpleNamespace(uid="graph-2"), "graph-2"),
        (object(), "object"),
    ],
)
def test_graph_identity_uses_stable_available_identifier(graph, expected):
    assert graph_identity(graph) == expected


@pytest.mark.parametrize(
    "fitness, expected",
    [
        (SimpleNamespace(values=(0.25,)), 0.25),
        (SimpleNamespace(value=0.5), 0.5),
        (SimpleNamespace(values=(float("inf"), 0.75)), 0.75),
        (SimpleNamespace(values=(None,)), None),
    ],
)
def test_fitness_scalar_returns_first_finite_value(fitness, expected):
    assert fitness_scalar(fitness) == expected
