from contextlib import nullcontext
import random
from types import SimpleNamespace

import pytest
from golem.core.optimisers.opt_history_objects.individual import Individual
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


class _UnlimitedTimer(nullcontext):
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
