from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fedot_ind.core.optimizer.domain import (
    CandidateFailureCode,
    EvaluationFailed,
    EvaluationFailure,
    EvaluationFailureCode,
    EvaluationOutcomeDecodeError,
    EvaluationReused,
    EvaluationSucceeded,
    EvolutionPhase,
    EvolutionState,
    EvolutionTransitionError,
    classify_candidate_failure,
    evaluation_outcome_from_metadata,
    evaluation_outcome_to_metadata,
    transition_evolution_state,
)


def test_candidate_failure_classification_has_stable_precedence():
    not_individual = classify_candidate_failure(
        is_individual=False,
        is_valid_graph=False,
        is_duplicate_graph=True,
    )
    verifier_rejection = classify_candidate_failure(
        is_individual=True,
        is_valid_graph=False,
        is_duplicate_graph=True,
        individual_id="individual",
        graph_id="graph",
    )
    duplicate = classify_candidate_failure(
        is_individual=True,
        is_valid_graph=True,
        is_duplicate_graph=True,
    )
    accepted = classify_candidate_failure(
        is_individual=True,
        is_valid_graph=True,
        is_duplicate_graph=False,
    )

    assert not_individual.code is CandidateFailureCode.MUTATION_DID_NOT_RETURN_INDIVIDUAL
    assert verifier_rejection.code is CandidateFailureCode.VERIFIER_REJECTED
    assert duplicate.code is CandidateFailureCode.DUPLICATE_GRAPH
    assert accepted is None
    assert verifier_rejection.to_record() == {
        "code": "verifier_rejected",
        "message": "Graph verifier rejected the candidate",
        "individual_id": "individual",
        "graph_id": "graph",
    }


@pytest.mark.parametrize("generation", [True, 1.5, -1])
def test_evolution_state_rejects_invalid_generation(generation):
    with pytest.raises(ValueError, match="generation"):
        EvolutionState(EvolutionPhase.CREATED, generation=generation)


def test_evolution_state_follows_the_supported_lifecycle():
    state = EvolutionState.created()
    state = transition_evolution_state(state, EvolutionPhase.INITIALISING)
    state = transition_evolution_state(state, EvolutionPhase.EVALUATING_INITIAL)
    state = transition_evolution_state(state, EvolutionPhase.EVOLVING)
    state = transition_evolution_state(state, EvolutionPhase.EVOLVING, generation=1)
    state = transition_evolution_state(state, EvolutionPhase.COMPLETED)

    assert state == EvolutionState(EvolutionPhase.COMPLETED, generation=1)


@pytest.mark.parametrize("terminal", [EvolutionPhase.COMPLETED, EvolutionPhase.FAILED])
def test_terminal_evolution_states_reject_every_transition(terminal):
    state = EvolutionState(terminal, generation=3)

    with pytest.raises(EvolutionTransitionError, match="transition is not allowed"):
        transition_evolution_state(state, EvolutionPhase.EVOLVING, generation=4)


def test_evolution_generation_cannot_move_backwards():
    state = EvolutionState(EvolutionPhase.EVOLVING, generation=3)

    with pytest.raises(EvolutionTransitionError, match="generation must be monotonic"):
        transition_evolution_state(state, EvolutionPhase.EVOLVING, generation=2)


@st.composite
def evaluation_outcomes(draw):
    duration = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    evaluated_at = draw(st.text(max_size=30))
    variant = draw(st.sampled_from(("succeeded", "failed", "reused")))
    if variant == "succeeded":
        return EvaluationSucceeded(duration_seconds=duration, evaluated_at=evaluated_at)
    if variant == "reused":
        return EvaluationReused(duration_seconds=duration)
    return EvaluationFailed(
        duration_seconds=duration,
        evaluated_at=evaluated_at,
        failure=EvaluationFailure(
            code=draw(st.sampled_from(tuple(EvaluationFailureCode))),
            error_type=draw(st.text(min_size=1, max_size=30)),
            message=draw(st.text(min_size=1, max_size=100)),
        ),
    )


@given(evaluation_outcomes())
def test_evaluation_outcome_metadata_round_trip(outcome):
    restored = evaluation_outcome_from_metadata(evaluation_outcome_to_metadata(outcome))

    assert restored == outcome


def test_legacy_evaluation_failure_metadata_remains_readable():
    outcome = evaluation_outcome_from_metadata({
        "computation_time_in_seconds": 0.5,
        "evaluation_time_iso": "2026-09-29T12:00:00",
        "evaluation_error": "bad graph",
        "evaluation_error_type": "ValueError",
    })

    assert isinstance(outcome, EvaluationFailed)
    assert outcome.failure.error_type == "ValueError"
    assert outcome.failure.message == "bad graph"


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"evaluation_schema": "other"}, "Unsupported evaluation schema"),
        ({"evaluation_status": "unknown"}, "Unsupported evaluation status"),
        ({"computation_time_in_seconds": "slow"}, "duration must be numeric"),
        ({
            "evaluation_status": "failed",
            "evaluation_failure": {
                "code": "unknown",
                "error_type": "ValueError",
                "message": "bad",
            },
        }, "failure code"),
    ],
)
def test_malformed_evaluation_metadata_fails_at_boundary(metadata, message):
    with pytest.raises(EvaluationOutcomeDecodeError, match=message):
        evaluation_outcome_from_metadata(metadata)
