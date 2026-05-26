from __future__ import annotations

from types import SimpleNamespace

from fedot_ind.api.services.fit import IndustrialFitService


class CallableStrategy:
    def __init__(self):
        self.fit_calls = []

    def __call__(self):
        return None

    def fit(self, data):
        self.fit_calls.append(data)
        return "strategy_result"


def test_fit_service_uses_callable_industrial_strategy():
    strategy = CallableStrategy()
    solver = SimpleNamespace(fit_calls=[])
    solver.fit = lambda data: solver.fit_calls.append(data)
    manager = SimpleNamespace(
        industrial_config=SimpleNamespace(strategy=strategy),
        solver=solver,
    )

    result = IndustrialFitService().fit(manager, "train")

    assert result == "strategy_result"
    assert strategy.fit_calls == ["train"]
    assert solver.fit_calls == []


def test_fit_service_uses_solver_when_strategy_is_not_callable():
    solver = SimpleNamespace(fit_calls=[])

    def fit(data):
        solver.fit_calls.append(data)
        return "solver_result"

    solver.fit = fit
    manager = SimpleNamespace(
        industrial_config=SimpleNamespace(strategy=None),
        solver=solver,
    )

    result = IndustrialFitService().fit(manager, "train")

    assert result == "solver_result"
    assert solver.fit_calls == ["train"]
