from __future__ import annotations

from types import SimpleNamespace

from fedot_ind.api.services.dask_runtime import DaskRuntimeInitializer
from fedot_ind.api.services.repository import IndustrialRepositoryInitializer
from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    MutationAgentType,
)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_repository_initializer_activates_default_fedot_context():
    class FakeSession:
        def activate(self):
            raise AssertionError("Default FEDOT context must not register Industrial operations.")

        def close(self):
            pass

    manager = SimpleNamespace(
        industrial_config=SimpleNamespace(is_default_fedot_context=True),
        automl_config=SimpleNamespace(optimisation_strategy={}),
        optimisation_agent={"Fedot": "fedot_optimizer"},
    )

    result = IndustrialRepositoryInitializer(FakeSession).activate(
        manager=manager,
        logger=FakeLogger(),
        input_data="input",
    )

    assert result.extension is None
    assert result.input_data == "input"
    assert manager.automl_config.optimisation_strategy == "fedot_optimizer"


def test_repository_initializer_activates_industrial_context_with_optimizer_partial():
    class FakeSession:
        def activate(self):
            return "registered"

        def close(self):
            pass

    def fake_optimizer(**kwargs):
        return kwargs

    manager = SimpleNamespace(
        industrial_config=SimpleNamespace(is_default_fedot_context=False),
        compute_config=SimpleNamespace(backend="cpu"),
        automl_config=SimpleNamespace(
            optimisation_strategy={
                "optimisation_agent": "Industrial",
                "optimisation_strategy": {"mutation_agent": "random"},
            }
        ),
        optimisation_agent={"Industrial": fake_optimizer},
    )

    result = IndustrialRepositoryInitializer(FakeSession).activate(
        manager=manager,
        logger=FakeLogger(),
    )

    assert result.extension == "registered"
    assert manager.automl_config.optimisation_strategy.func is fake_optimizer
    optimisation_params = manager.automl_config.optimisation_strategy.keywords["optimisation_params"]
    assert isinstance(optimisation_params, EvolutionConfig)
    assert optimisation_params.mutation_agent is MutationAgentType.RANDOM
    assert optimisation_params.execution_policy.initial_graphs_prevalidated is True


def test_repository_initializer_activation_is_idempotent_and_close_is_owned():
    calls = []

    class FakeSession:
        def activate(self):
            calls.append("activate")
            return "registered"

        def close(self):
            calls.append("close")

    initializer = IndustrialRepositoryInitializer(FakeSession)

    assert initializer.ensure_active(industrial_context=False) is None
    assert initializer.ensure_active(industrial_context=True) == "registered"
    initializer.close()
    assert calls == ["activate", "close"]


def test_dask_runtime_initializer_returns_client_and_cluster_handles():
    class FakeClient:
        dashboard_link = "http://dask"

    class FakeDaskServer:
        def __init__(self, distributed_config):
            self.distributed_config = distributed_config
            self.client = FakeClient()
            self.cluster = "cluster"

    runtime = DaskRuntimeInitializer(FakeDaskServer).start(
        distributed_config={"n_workers": 1},
        logger=FakeLogger(),
    )

    assert runtime.client.dashboard_link == "http://dask"
    assert runtime.cluster == "cluster"
