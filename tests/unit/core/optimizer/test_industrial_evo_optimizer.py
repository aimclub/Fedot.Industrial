from types import SimpleNamespace
from contextlib import nullcontext
from datetime import timedelta
import random

import pytest
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot_ind.core.optimizer.IndustrialEvoOptimizer import (
    IndustrialEvoOptimizer,
    IndustrialPopulationError,
)
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.core.optimizer.domain import EvolutionPhase
from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy, ResourceBudget
from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier
from fedot_ind.core.optimizer.randomness import CandidateRandomness


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
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    graph_optimizer_params = SimpleNamespace(
        mutation_types=[MutationTypesEnum.single_change],
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


@pytest.mark.parametrize("seed", [None, 42])
def test_adaptive_agent_learning_uses_its_own_seed_sequence(seed):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    population = [Individual(_Graph("initial"))]
    draws = []
    optimizer.evolution_config = EvolutionConfig(random_seed=seed)
    optimizer._agent_learning_attempt = 0
    optimizer.regularization = lambda pop, evaluator: pop
    optimizer.reproducer = SimpleNamespace(reproduce=lambda pop, evaluator: pop, stop_condition=False)
    optimizer.mutation = SimpleNamespace(
        agent_experience=SimpleNamespace(collect_results=lambda pop: None),
        agent=SimpleNamespace(partial_fit=lambda experience: draws.append(random.random())),
    )
    optimizer.inheritance = lambda previous, new: new
    optimizer.elitism = lambda best, new: new
    optimizer.generations = SimpleNamespace(best_individuals=population)
    optimizer.log = SimpleNamespace(info=lambda message: None)
    state = random.getstate()
    try:
        assert optimizer._evolve_population(population, "evaluator") == (population, "evaluator")
        optimizer._evolve_population(population, "evaluator")
        if seed is None:
            expected = random.Random()
            expected.setstate(state)
            assert draws == [expected.random(), expected.random()]
            assert optimizer._agent_learning_attempt == 0
        else:
            assert draws == [random.Random(CandidateRandomness(seed, "agent_learning", i).seed).random()
                             for i in range(2)]
            assert random.getstate() == state
            assert optimizer._agent_learning_attempt == 2
    finally:
        random.setstate(state)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("execution_backend", ["legacy", "sequential", "process"])
def test_real_optimizer_completes_generation_with_builtin_graph_objective(
        monkeypatch, tmp_path, enabled, execution_backend):
    """Exercise real GOLEM reproduction through each supported execution boundary."""
    import dask
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from fedot_ind.core.optimizer.configuration import ExecutionPolicy, MutationStrategy
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
    from fedot_ind.core.repository import IndustrialDispatcher as dispatcher_module

    monkeypatch.setattr(dispatcher_module, "industrial_extension_scope", nullcontext)
    requirements, params, graph = _environment(
        {"name": "test", "task": "regression", "operations": ["scaling", "normalization", "ridge", "dtreg"],
         "initial_chain": ["scaling", "ridge"]}, {"max_depth": 4, "max_arity": 3})
    requirements.num_of_generations = 2
    # This checks completed generations, not spawn latency on the host running pytest.
    # Deadline enforcement has separate executor tests with bounded objectives.
    requirements.timeout = timedelta(minutes=3)
    requirements.agent_dir = str(tmp_path)
    objective = Objective({"size": lambda pipeline: float(len(pipeline.nodes))})
    optimizer = IndustrialEvoOptimizer(
        objective, [graph], requirements, params,
        GPAlgorithmParameters(pop_size=2, max_pop_size=3, mutation_prob=1.0,
                              mutation_types=[MutationTypesEnum.single_add, MutationTypesEnum.single_change]),
        EvolutionConfig(random_seed=42, mutation_strategy=MutationStrategy.GROWTH,
                        evaluation_execution=EvaluationExecutionConfig(EvaluationBackend(execution_backend)),
                        mutation_planning=MutationPlanningPolicy(enabled=enabled),
                        resource_budget=ResourceBudget(2, 2, 8),
                        execution_policy=ExecutionPolicy(diagnostics_output_dir=str(tmp_path / "diagnostics"))))
    with dask.config.set(scheduler="synchronous"):
        result = optimizer.optimise(objective)
    assert result
    assert optimizer.evolution_state.phase is EvolutionPhase.COMPLETED
    assert optimizer.diagnostics.summary.evaluations_succeeded > 0
    assert optimizer.diagnostics.summary.population_observations >= 2
    assert (tmp_path / "diagnostics" / "evolution_summary.json").is_file()
    if enabled:
        assert any(event.event_type.value == "mutation_plan" for event in optimizer.diagnostics.observations)


def test_probability_strategy_restores_random_agent_even_when_extension_fails():
    from golem.core.optimisers.adaptive.operator_agent import RandomAgent
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    agent = RandomAgent([MutationTypesEnum.single_add, MutationTypesEnum.single_change], [.5, .5])
    optimizer = IndustrialEvoOptimizer.__new__(IndustrialEvoOptimizer)
    optimizer.mutation = SimpleNamespace(agent=agent)
    with pytest.raises(RuntimeError):
        with optimizer._mutation_probability_strategy("initial_population_diversity_strategy"):
            raise RuntimeError("failed extension")
    assert agent._probs == [.5, .5]


def test_probability_strategy_does_not_override_adaptive_agent():
    optimizer = IndustrialEvoOptimizer.__new__(IndustrialEvoOptimizer)
    optimizer.mutation = SimpleNamespace(agent=SimpleNamespace())
    with optimizer._mutation_probability_strategy("initial_population_diversity_strategy"):
        assert not hasattr(optimizer.mutation.agent, "_probs")


def test_population_extension_does_not_repeat_an_exhausted_plan_pool():
    from fedot_ind.core.optimizer.mutation_runtime import MutationExecution, MutationExecutionStatus
    parent = Individual(_Graph("initial"))
    calls = []

    class Exhausted:
        last_execution = MutationExecution(MutationExecutionStatus.NO_ADMISSIBLE_PLAN)

        def __call__(self, individual):
            calls.append(individual)
            return []

    optimizer = _observable_optimizer((), verifier=lambda graph: True)
    optimizer.graph_generation_attempts = 100
    optimizer.min_reproduce_attempt = 50
    optimizer.mutation = Exhausted()
    assert optimizer._extend_population([parent], 2) == [parent]
    assert calls == [parent]


def test_population_extension_obeys_shared_timer_before_mutation():
    parent = Individual(_Graph("initial"))
    optimizer = _observable_optimizer((Individual(_Graph("new")),), verifier=lambda graph: True)
    optimizer.timer = SimpleNamespace(is_time_limit_reached=lambda: True)
    assert optimizer._extend_population([parent], 2) == [parent]
    assert optimizer.diagnostics.summary.mutation_attempts == 0
