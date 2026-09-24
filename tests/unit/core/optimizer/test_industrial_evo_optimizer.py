from types import SimpleNamespace

import pytest

from fedot_ind.core.optimizer.IndustrialEvoOptimizer import (
    IndustrialEvoOptimizer,
    IndustrialPopulationError,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _optimizer_with_failed_initial_population(*, prevalidated: bool):
    optimizer = object.__new__(IndustrialEvoOptimizer)
    optimizer.timer = _Context()
    optimizer.requirements = SimpleNamespace(show_progress=False, num_of_generations=1)
    optimizer.eval_dispatcher = SimpleNamespace(dispatch=lambda objective, timer: "evaluator")
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


def test_timeout_without_prevalidation_preserves_population_error():
    optimizer = _optimizer_with_failed_initial_population(prevalidated=False)

    with pytest.raises(IndustrialPopulationError, match="evaluation failed"):
        optimizer.optimise(object())
