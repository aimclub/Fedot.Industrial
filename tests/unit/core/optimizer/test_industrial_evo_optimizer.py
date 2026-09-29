from types import SimpleNamespace

import pytest
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.optimizer.IndustrialEvoOptimizer import (
    IndustrialEvoOptimizer,
    IndustrialPopulationError,
)
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.core.optimizer.domain import EvolutionPhase
from fedot_ind.core.optimizer.configuration import EvolutionConfig, ResourceBudget
from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _optimizer_with_failed_initial_population(*, prevalidated: bool):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    optimizer.timer = _Context()
    optimizer.requirements = SimpleNamespace(
        show_progress=False, num_of_generations=1)
    optimizer.eval_dispatcher = SimpleNamespace(
        dispatch=lambda objective, timer: "evaluator")
    optimizer.initial_graphs = ["initial_graph"]
    optimizer.initial_graphs_prevalidated = prevalidated
    optimizer.log = SimpleNamespace(warning=lambda message: None)

    def fail_initial_population(evaluator):
        del evaluator
        raise IndustrialPopulationError("evaluation failed", timed_out=True)

    optimizer._initial_population = fail_initial_population
    return optimizer


def test_timeout_returns_initial_graph_only_when_fedot_prevalidated_it():
    optimizer = _optimizer_with_failed_initial_population(prevalidated=True)

    result = optimizer.optimise(object())

    assert result == ["initial_graph"]
    assert optimizer.evolution_state.phase is EvolutionPhase.COMPLETED


def test_timeout_without_prevalidation_preserves_population_error():
    optimizer = _optimizer_with_failed_initial_population(prevalidated=False)

    with pytest.raises(IndustrialPopulationError, match="evaluation failed"):
        optimizer.optimise(object())

    assert optimizer.evolution_state.phase is EvolutionPhase.FAILED


class _Graph:
    def __init__(self, graph_id):
        self.descriptive_id = graph_id

    def __eq__(self, other):
        return isinstance(other, _Graph) and self.descriptive_id == other.descriptive_id


def _observable_optimizer(mutation_results, verifier):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    optimizer.graph_generation_attempts = len(mutation_results)
    optimizer.min_reproduce_attempt = 1
    optimizer.graph_generation_params = SimpleNamespace(verifier=verifier)
    optimizer.diagnostics_recorder = EvolutionDiagnosticsRecorder()
    results = iter(mutation_results)
    optimizer.mutation = lambda individual: next(results)
    return optimizer


def test_optimizer_returns_empty_diagnostics_when_collection_is_disabled():
    optimizer = IndustrialEvoOptimizer.__new__(IndustrialEvoOptimizer)
    optimizer.diagnostics_recorder = None

    diagnostics = optimizer.diagnostics

    assert diagnostics.observations == ()
    assert diagnostics.summary.mutation_attempts == 0


def test_optimizer_uses_typed_resource_budget_for_internal_limits():
    optimizer = IndustrialEvoOptimizer.__new__(IndustrialEvoOptimizer)
    graph_optimizer_params = SimpleNamespace(
        mutation_types=[],
        adaptive_mutation_type=None,
    )
    config = EvolutionConfig(
        resource_budget=ResourceBudget(
            min_population_size=3,
            mutation_attempts_per_candidate=5,
            population_extension_attempts=7,
        )
    )

    optimizer._init_industrial_optimizer_params(graph_optimizer_params, config)

    assert optimizer.min_pop_size == 3
    assert optimizer.min_reproduce_attempt == 5
    assert optimizer.graph_generation_attempts == 7


def test_population_extension_records_attempts_acceptance_and_duplicate_rejection():
    initial = Individual(_Graph("initial"))
    accepted = Individual(_Graph("accepted"))
    duplicate = Individual(_Graph("accepted"))
    optimizer = _observable_optimizer(
        (accepted, duplicate), verifier=lambda graph: True)

    population = optimizer._extend_population([initial], target_pop_size=3)
    summary = optimizer.diagnostics.summary

    assert population == [initial, accepted]
    assert summary.mutation_attempts == 2
    assert summary.candidates_accepted == 1
    assert summary.candidates_rejected == 1
    assert summary.rejection_counts == {"duplicate_graph": 1}
    assert summary.valid_offspring_ratio == 0.5


def test_population_extension_records_verifier_rejection():
    initial = Individual(_Graph("initial"))
    rejected = Individual(_Graph("rejected"))
    optimizer = _observable_optimizer(
        (rejected,), verifier=lambda graph: False)

    population = optimizer._extend_population([initial], target_pop_size=2)

    assert population == [initial]
    assert optimizer.diagnostics.summary.rejection_counts == {
        "verifier_rejected": 1}


def test_population_extension_retains_structured_graph_validation_reasons():
    initial = Individual(_Graph("initial"))
    rejected = Individual(_Graph("rejected"))
    verifier = IndustrialGraphVerifier(adapter=None, task_type="classification")
    optimizer = _observable_optimizer((rejected,), verifier=verifier)

    population = optimizer._extend_population([initial], target_pop_size=2)

    assert population == [initial]
    assert optimizer.diagnostics.summary.validation_issue_counts == {
        "empty_graph": 1,
    }
    candidate_event = optimizer.diagnostics.observations[-1]
    assert candidate_event.validation_issue_codes == ("empty_graph",)


def test_population_extension_records_missing_individual_without_string_sentinel():
    initial = Individual(_Graph("initial"))
    optimizer = _observable_optimizer((None,), verifier=lambda graph: True)

    population = optimizer._extend_population([initial], target_pop_size=2)

    assert population == [initial]
    assert optimizer.diagnostics.summary.rejection_counts == {
        "mutation_did_not_return_individual": 1,
    }


def test_population_extension_records_verifier_error_before_preserving_exception():
    initial = Individual(_Graph("initial"))
    candidate = Individual(_Graph("candidate"))

    def fail_verification(graph):
        raise ValueError(f"cannot verify {graph.descriptive_id}")

    optimizer = _observable_optimizer((candidate,), verifier=fail_verification)

    with pytest.raises(ValueError, match="cannot verify candidate"):
        optimizer._extend_population([initial], target_pop_size=2)

    summary = optimizer.diagnostics.summary
    assert summary.rejection_counts == {"verifier_failed": 1}
