from __future__ import annotations

from types import SimpleNamespace

from fedot_ind.api.main import FedotIndustrial


def test_fit_orchestrates_processing_repository_solver_and_fit_in_order():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace()
    industrial.shutdown = lambda: calls.append("shutdown")

    def process(input_data):
        calls.append(("process", input_data))
        return "processed"

    def init_backend(data):
        calls.append(("repository", data))
        return "repository_ready"

    def init_solver(data):
        calls.append(("solver", data))
        return "solver_ready"

    class FakeFitService:
        def fit(self, manager, train_data):
            calls.append(("fit", manager, train_data))

    industrial._process_input_data = process
    industrial._FedotIndustrial__init_industrial_backend = init_backend
    industrial._FedotIndustrial__init_solver = init_solver
    industrial.fit_service = FakeFitService()

    industrial.fit("raw")

    assert calls == [
        ("process", "raw"),
        ("repository", "processed"),
        ("solver", "repository_ready"),
        ("fit", industrial.manager, "solver_ready"),
    ]


def test_predict_orchestrates_repository_processing_and_prediction():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace(
        compute_config=SimpleNamespace(backend="cpu"),
    )

    class FakeRepositoryInitializer:
        def setup_repository(self, backend):
            calls.append(("repository", backend))
            return "repo"

    industrial.repository_initializer = FakeRepositoryInitializer()
    industrial._process_input_data = lambda data: calls.append(("process", data)) or "processed"
    industrial._FedotIndustrial__abstract_predict = (
        lambda data, mode: calls.append(("predict", data, mode)) or "labels"
    )

    result = industrial.predict("raw", predict_mode="labels")

    assert result == "labels"
    assert industrial.repo == "repo"
    assert industrial.manager.predict_data == "processed"
    assert industrial.manager.predicted_labels == "labels"
    assert calls == [
        ("repository", "cpu"),
        ("process", "raw"),
        ("predict", "processed", "labels"),
    ]


def test_predict_proba_for_regression_uses_label_mode():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace(
        compute_config=SimpleNamespace(backend="cpu"),
        industrial_config=SimpleNamespace(is_regression_task_context=True),
    )

    class FakeRepositoryInitializer:
        def setup_repository(self, backend):
            calls.append(("repository", backend))
            return "repo"

    industrial.repository_initializer = FakeRepositoryInitializer()
    industrial._process_input_data = lambda data: calls.append(("process", data)) or "processed"
    industrial._FedotIndustrial__abstract_predict = (
        lambda data, mode: calls.append(("predict", data, mode)) or "predicted"
    )

    result = industrial.predict_proba("raw", predict_mode="probs")

    assert result == "predicted"
    assert industrial.manager.predicted_probs == "predicted"
    assert calls == [
        ("repository", "cpu"),
        ("process", "raw"),
        ("predict", "processed", "labels"),
    ]
