from datetime import timedelta
from copy import deepcopy
import random
from types import SimpleNamespace

import pytest
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.timer import OptimisationTimer
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from fedot_ind.core.optimizer.configuration import EvolutionConfig
from fedot_ind.core.optimizer.domain import (
    EvolutionPhase, EvolutionState, EvolutionTransitionError, transition_evolution_state,
    IndustrialPopulationError,
)
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.optimizer.observability import EvolutionDiagnosticsRecorder
from fedot_ind.core.optimizer.randomness import SeededMutation


# Independent lifecycle oracle: deliberately not imported from production tables.
_NEXT = {
    "created": {"initialising", "failed"},
    "initialising": {"evaluating_initial", "failed"},
    "evaluating_initial": {"evolving", "completed", "failed"},
    "evolving": {"evolving", "completed", "failed"},
    "completed": set(), "failed": set(),
}


class EvolutionLifecycleMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.actual = EvolutionState.created()
        self.phase, self.generation = "created", 0

    @rule(next_phase=st.sampled_from(list(EvolutionPhase)), generation=st.integers(min_value=0, max_value=30))
    def transition(self, next_phase, generation):
        target = next_phase
        before = self.actual
        allowed = target.value in _NEXT[self.phase] and generation >= self.generation
        if allowed:
            self.actual = transition_evolution_state(self.actual, target, generation=generation)
            self.phase, self.generation = target.value, generation
        else:
            with pytest.raises(EvolutionTransitionError):
                transition_evolution_state(self.actual, target, generation=generation)
            assert self.actual is before

    @invariant()
    def state_matches_model(self):
        assert self.actual.phase.value == self.phase
        assert self.actual.generation == self.generation

    @precondition(lambda self: bool(_NEXT[self.phase]))
    @rule(data=st.data(), increment=st.integers(min_value=0, max_value=5))
    def legal_step(self, data, increment):
        phase = data.draw(st.sampled_from(sorted(_NEXT[self.phase])))
        self.transition(EvolutionPhase(phase), self.generation + increment)

    @precondition(lambda self: not _NEXT[self.phase])
    @rule()
    def new_run(self):
        self.actual = EvolutionState.created()
        self.phase, self.generation = "created", 0


TestEvolutionLifecycle = EvolutionLifecycleMachine.TestCase
TestEvolutionLifecycle.settings = settings(max_examples=60, stateful_step_count=40, deadline=None)


class _Graph:
    def __init__(self, name):
        self.descriptive_id = name

    def __eq__(self, other):
        return isinstance(other, _Graph) and self.descriptive_id == other.descriptive_id


class _UnlimitedTimer(OptimisationTimer):
    def is_time_limit_reached(self):
        return False


class _LifecycleHarness(IndustrialEvoOptimizer):
    """Keep the production loop, replacing only evaluation and generation effects."""

    @property
    def current_generation_num(self):
        return self.generation

    @property
    def best_individuals(self):
        return self.population

    @property
    def _progressbar(self):
        return self.progress

    def _update_population(self, next_population, evaluator=None, label=None, metadata=None):
        self.population = next_population
        self.labels.append(label)
        if self.evolution_state.phase is EvolutionPhase.EVOLVING:
            self._transition_evolution(EvolutionPhase.EVOLVING, generation=self.generation)
        return next_population

    def _initial_population(self, evaluator):
        self._update_population(evaluator(self.population), label="initial")
        return self.population, evaluator

    def _optimise_loop(self, population_to_eval, evaluator):
        population = self._extend_population(population_to_eval, target_pop_size=2)
        self.generation += 1
        return self._update_population(evaluator(population), label="evolved")


def _harness(seed, mutation):
    optimizer = object.__new__(_LifecycleHarness)
    optimizer.evolution_config = EvolutionConfig(random_seed=seed)
    optimizer.evolution_state = EvolutionState.created()
    optimizer._parent_selection_attempt = 0
    optimizer.timer = _UnlimitedTimer()
    optimizer.generation = 0
    optimizer.population = [Individual(_Graph("initial"))]
    optimizer.labels = []
    optimizer.evaluated_population = []
    optimizer.eval_dispatcher = SimpleNamespace(dispatch=lambda objective, timer: objective)
    optimizer.progress = SimpleNamespace(update=lambda: None, close=lambda: None)
    optimizer.stop_optimization = lambda: optimizer.generation >= 1
    optimizer.graph_generation_params = SimpleNamespace(verifier=lambda graph: True)
    optimizer.graph_generation_attempts = 4
    optimizer.min_reproduce_attempt = 1
    optimizer.diagnostics_recorder = EvolutionDiagnosticsRecorder()
    optimizer.diagnostics_output_dir = None
    optimizer.mutation = SeededMutation(mutation, seed, "mutation")
    return optimizer


def test_public_loop_rejects_duplicate_finishes_budget_and_replays_decisions():
    def run():
        attempt = 0

        def mutate(parent):
            nonlocal attempt
            attempt += 1
            return Individual(_Graph("initial" if attempt == 1 else "accepted"))

        optimizer = _harness(42, mutate)
        result = optimizer.optimise(lambda population: population)
        assert optimizer.evolution_state == EvolutionState(EvolutionPhase.COMPLETED, 1)
        assert optimizer.labels == ["initial", "evolved", "final_choices"]
        assert optimizer.diagnostics.summary.rejection_counts == {"duplicate_graph": 1}
        assert optimizer.diagnostics.summary.candidates_accepted == 1
        assert optimizer.mutation.attempt == 2
        assert optimizer._parent_selection_attempt == 2
        with pytest.raises(EvolutionTransitionError):
            optimizer.optimise(lambda population: population)
        assert optimizer.evolution_state.phase is EvolutionPhase.COMPLETED
        return [(g.descriptive_id) for g in result], optimizer.population[-1].metadata

    assert run() == run()


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_public_loop_preserves_failure_and_closes_progress(error_type):
    def fail(parent):
        raise error_type("operator failure")

    optimizer = _harness(42, fail)
    closed, exported = [], []
    optimizer.progress.close = lambda: closed.append(True)
    optimizer.diagnostics_output_dir = "requested-report"
    optimizer.export_diagnostics = lambda path: exported.append(path)
    with pytest.raises(error_type, match="operator failure"):
        optimizer.optimise(lambda population: population)
    assert optimizer.evolution_state.phase is EvolutionPhase.FAILED
    assert closed == [True]
    assert exported == ["requested-report"]


@pytest.mark.parametrize('scores', [(0.2, 0.9), (0.9, 0.2)])
def test_initial_candidates_own_metadata_and_keep_their_evaluation_identity(scores):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    template = {'source': {'labels': ['frozen']}}
    original = deepcopy(template)
    optimizer.requirements = SimpleNamespace(static_individual_metadata=template)
    optimizer.graph_optimizer_params = SimpleNamespace(pop_size=2)
    optimizer._admissible_initial_graphs = lambda: [_Graph('first'), _Graph('second')]
    optimizer._update_population = lambda **kwargs: None

    def evaluate(population):
        for individual, score in zip(population, scores):
            individual.metadata.update(evaluation_context={'candidate_id': individual.uid, 'fitness': score})
            individual.metadata['source']['labels'].append(individual.graph.descriptive_id)
            individual.set_evaluation_result(SingleObjFitness(score))
        return population

    population, _ = optimizer._initial_population(evaluate)
    assert template == original
    first, second = population
    assert first.metadata is not second.metadata
    assert first.metadata['source'] is not second.metadata['source']
    for individual, score in zip(population, scores):
        assert individual.metadata['evaluation_context'] == {'candidate_id': individual.uid, 'fitness': score}
        assert individual.metadata['source']['labels'] == ['frozen', individual.graph.descriptive_id]


class _ControlledTimer(OptimisationTimer):
    """Run the real GOLEM stop calculation against a manually advanced clock."""

    def __init__(self, timeout):
        super().__init__(timeout=timeout)
        self.elapsed_seconds = 0
        self.initial_boundaries = []
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        super().__enter__()
        self.original_start = self.start_time
        return self

    @property
    def spent_time(self):
        return timedelta(seconds=self.elapsed_seconds)

    def set_init_time(self, init_time):
        self.initial_boundaries.append((init_time, self.start_time))
        super().set_init_time(init_time)


def _timed_harness(initial_seconds, generation_seconds, *, timeout=120):
    optimizer = _harness(42, lambda parent: Individual(_Graph("candidate")))
    timer = _ControlledTimer(None if timeout is None else timedelta(seconds=timeout))
    optimizer.timer = timer
    initial_population = optimizer._initial_population
    iterations = iter(generation_seconds)

    def initialise(evaluator):
        result = initial_population(evaluator)
        timer.elapsed_seconds += initial_seconds
        optimizer.generation = 1  # Native GOLEM counts the evaluated initial population.
        return result

    def evolve(population_to_eval, evaluator):
        timer.elapsed_seconds += next(iterations)
        optimizer.generation += 1
        return optimizer._update_population(population_to_eval, label="evolved")

    optimizer._initial_population = initialise
    optimizer._optimise_loop = evolve
    optimizer.stop_optimization = lambda: (
        timer.is_time_limit_reached(optimizer.generation - 1) or optimizer.generation >= 4)
    return optimizer, timer


def test_initial_population_time_is_not_reused_as_generation_duration():
    optimizer, timer = _timed_harness(40, [32, 16, 6])
    optimizer.optimise(lambda population: population)

    # The broken estimate was 72 + 72 > 120; the correct one is 72 + 32 < 120.
    assert optimizer.labels == ["initial", "evolved", "evolved", "evolved", "final_choices"]
    assert timer.initial_boundaries == [(40 / 60, timer.start_time)]
    assert timer.start_time == timer.original_start
    assert timer.entries == 1
    assert timer.seconds_from_start == 94
    assert timer.timeout == timedelta(seconds=120)
    assert not timer.process_terminated


@pytest.mark.parametrize("initial_seconds", [120, 121])
def test_initial_population_still_consumes_the_total_deadline(initial_seconds):
    optimizer, timer = _timed_harness(initial_seconds, [])
    optimizer.optimise(lambda population: population)

    assert optimizer.labels == ["initial", "final_choices"]
    assert timer.initial_boundaries == [(initial_seconds / 60, timer.start_time)]
    assert timer.start_time == timer.original_start
    assert timer.entries == 1
    assert timer.process_terminated
    assert len(optimizer.evaluated_population) == 1


@pytest.mark.parametrize("initial, duration", [(60, 40), (40, 40)])
def test_next_generation_prediction_still_stops_before_actual_deadline(initial, duration):
    optimizer, timer = _timed_harness(initial, [duration])
    optimizer.optimise(lambda population: population)

    assert timer.seconds_from_start == initial + duration
    assert timer.seconds_from_start < timer.timeout.total_seconds()
    assert timer.process_terminated  # Including equality: 80 + 40 cannot fit strictly before 120.
    assert optimizer.labels == ["initial", "evolved", "final_choices"]
    assert len(timer.initial_boundaries) == 1


def test_time_marker_after_all_generations_is_not_a_missing_evolution_step():
    optimizer, timer = _timed_harness(40, [32, 16, 12])
    optimizer.optimise(lambda population: population)
    assert timer.process_terminated
    assert optimizer.labels.count("evolved") == 3
    assert len(optimizer.evaluated_population) == 4
    assert timer.start_time == timer.original_start
    assert timer.entries == 1


def test_unlimited_timer_uses_the_same_initial_boundary_without_a_deadline():
    optimizer, timer = _timed_harness(300, [300, 300, 300], timeout=None)
    optimizer.optimise(lambda population: population)

    assert timer.initial_boundaries == [(5, timer.start_time)]
    assert optimizer.labels.count("evolved") == 3
    assert not timer.process_terminated


def test_failed_initial_population_does_not_publish_an_evolution_boundary():
    optimizer, timer = _timed_harness(40, [])

    def fail(evaluator):
        raise IndustrialPopulationError("initial evaluation failed", timed_out=False)

    optimizer._initial_population = fail
    with pytest.raises(IndustrialPopulationError, match="initial evaluation failed"):
        optimizer.optimise(lambda population: population)
    assert timer.initial_boundaries == []
    assert optimizer.evolution_state.phase is EvolutionPhase.FAILED


def test_native_initial_expansion_and_generation_keeper_share_the_clock(tmp_path):
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.objective import Objective
    from benchmark.industrial.experiments.mutation_planning.runner import _environment

    requirements, params, graph = _environment(
        {"name": "clock", "task": "regression", "operations": ["scaling", "ridge"],
         "initial_chain": ["scaling", "ridge"]}, {"max_depth": 4, "max_arity": 2})
    requirements.num_of_generations = 3
    requirements.early_stopping_iterations = 100
    requirements.timeout = timedelta(seconds=120)
    requirements.agent_dir = str(tmp_path)
    optimizer = IndustrialEvoOptimizer(
        Objective({"size": len}), [graph], requirements, params,
        GPAlgorithmParameters(pop_size=2, max_pop_size=2, mutation_prob=1.,
                              mutation_types=[MutationTypesEnum.single_drop]))
    timer = _ControlledTimer(requirements.timeout)
    optimizer.timer = timer
    callbacks = []

    class Mutation:
        agent = SimpleNamespace()

        def __call__(self, parent):
            from golem.core.optimisers.graph import OptGraph, OptNode
            timer.elapsed_seconds += 10
            return Individual(OptGraph(OptNode("ridge")))

    def evaluate(population):
        timer.elapsed_seconds += 20
        for individual in population:
            individual.set_evaluation_result(SingleObjFitness(1.))
        return population

    def observe(population, owner):
        callbacks.append((owner.current_generation_num, bool(owner.final_individuals)))
        if owner.current_generation_num == 1:
            timer.elapsed_seconds += 10

    def evolve(population_to_eval, evaluator):
        timer.elapsed_seconds += 40
        return optimizer._update_population(population_to_eval, evaluator=evaluator)

    optimizer.mutation = Mutation()
    optimizer.eval_dispatcher = SimpleNamespace(dispatch=lambda objective, clock: evaluate)
    optimizer._optimise_loop = evolve
    optimizer.set_iteration_callback(observe)
    optimizer.optimise(object())

    assert callbacks == [(1, False), (2, False), (2, True)]  # Final choice is not another generation.
    assert optimizer.current_generation_num == 2
    assert timer.initial_boundaries == [(40 / 60, timer.original_start)]
    assert timer.start_time == timer.original_start
    assert timer.seconds_from_start == 80
    assert timer.process_terminated  # 80 + 40 equals the original deadline.
    assert len(optimizer.evaluated_population) == 2
    assert len(optimizer.evaluated_population[0]) == 2


def test_exhausted_attempt_budget_terminates_without_fabricated_candidates():
    optimizer = _harness(42, lambda parent: [])
    result = optimizer.optimise(lambda population: population)
    assert [graph.descriptive_id for graph in result] == ["initial"]
    assert optimizer.evolution_state.phase is EvolutionPhase.COMPLETED
    assert optimizer.mutation.attempt == 4
    assert optimizer.diagnostics.summary.candidates_rejected == 4


def test_loop_replays_stochastic_candidates_independent_of_ambient_state():
    def run(ambient_seed, run_seed):
        random.seed(ambient_seed)
        optimizer = _harness(run_seed, lambda parent: Individual(_Graph(str(random.randrange(10**9)))))
        optimizer.optimise(lambda population: population)
        return [(individual.graph.descriptive_id, individual.metadata) for individual in optimizer.population]

    state = random.getstate()
    try:
        assert run(10, 42) == run(20, 42)
        assert run(10, 42) != run(10, 43)
    finally:
        random.setstate(state)


def test_satisfied_population_budget_does_not_invoke_mutation():
    optimizer = _harness(42, lambda parent: pytest.fail("unexpected mutation"))
    assert optimizer._extend_population(optimizer.population, 1) == optimizer.population
    assert optimizer.mutation.attempt == 0


def test_population_extension_requires_a_parent_when_budget_is_not_satisfied():
    optimizer = _harness(42, lambda parent: pytest.fail("unexpected mutation"))
    with pytest.raises(IndustrialPopulationError, match="empty population"):
        optimizer._extend_population([], 1)
    assert optimizer._extend_population([], 0) == []
    assert optimizer.mutation.attempt == 0


@pytest.mark.parametrize("phase", ["created", None, 42])
def test_state_rejects_untyped_phase(phase):
    with pytest.raises(ValueError, match="EvolutionPhase"):
        EvolutionState(phase)
