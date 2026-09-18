from __future__ import annotations

from types import SimpleNamespace

from fedot_ind.api.services.dask_runtime import DaskRuntimeInitializer
from fedot_ind.api.services.repository import IndustrialRepositoryInitializer


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_repository_initializer_activates_default_fedot_context():
    class FakeIndustrialModels:
        def setup_default_repository(self):
            return "default_repo"

    manager = SimpleNamespace(
        industrial_config=SimpleNamespace(is_default_fedot_context=True),
        automl_config=SimpleNamespace(optimisation_strategy={}),
        optimisation_agent={"Fedot": "fedot_optimizer"},
    )

    result = IndustrialRepositoryInitializer(lambda: FakeIndustrialModels()).activate(
        manager=manager,
        logger=FakeLogger(),
        input_data="input",
    )

    assert result.repo == "default_repo"
    assert result.input_data == "input"
    assert manager.automl_config.optimisation_strategy == "fedot_optimizer"


def test_repository_initializer_activates_industrial_context_with_optimizer_partial():
    class FakeIndustrialModels:
        def setup_repository(self, backend):
            return f"industrial_repo:{backend}"

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

    result = IndustrialRepositoryInitializer(lambda: FakeIndustrialModels()).activate(
        manager=manager,
        logger=FakeLogger(),
    )

    assert result.repo == "industrial_repo:cpu"
    assert manager.automl_config.optimisation_strategy.func is fake_optimizer
    assert manager.automl_config.optimisation_strategy.keywords == {
        "optimisation_params": {"mutation_agent": "random"}
    }


def test_repository_initializer_can_setup_repository_without_optimizer_mutation():
    class FakeIndustrialModels:
        def setup_repository(self, backend):
            return f"repo:{backend}"

    repo = IndustrialRepositoryInitializer(lambda: FakeIndustrialModels()).setup_repository(backend="gpu")

    assert repo == "repo:gpu"


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
