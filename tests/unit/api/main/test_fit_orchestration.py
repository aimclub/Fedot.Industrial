from __future__ import annotations

from types import SimpleNamespace

import pytest

from fedot_ind.api.main import FedotIndustrial


def test_fit_orchestrates_processing_repository_solver_and_fit_in_order():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace()
    industrial.shutdown = lambda: calls.append("shutdown")

    def process(input_data, *, fit_stage):
        calls.append(("process", input_data, fit_stage))
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
        ("process", "raw", True),
        ("repository", "processed"),
        ("solver", "repository_ready"),
        ("fit", industrial.manager, "solver_ready"),
    ]


def test_predict_orchestrates_processing_and_prediction_without_repository_setup():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace(
        compute_config=SimpleNamespace(backend="cpu"),
    )

    industrial._process_input_data = (
        lambda data, *, fit_stage: calls.append(
            ("process", data, fit_stage)) or "processed"
    )
    industrial._FedotIndustrial__abstract_predict = (
        lambda data, mode: calls.append(("predict", data, mode)) or "labels"
    )

    result = industrial.predict("raw", predict_mode="labels")

    assert result == "labels"
    assert industrial.manager.predict_data == "processed"
    assert industrial.manager.predicted_labels == "labels"
    assert calls == [
        ("process", "raw", False),
        ("predict", "processed", "labels"),
    ]


def test_predict_proba_for_regression_uses_label_mode():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.manager = SimpleNamespace(
        compute_config=SimpleNamespace(backend="cpu"),
        industrial_config=SimpleNamespace(is_regression_task_context=True),
    )

    industrial._process_input_data = (
        lambda data, *, fit_stage: calls.append(
            ("process", data, fit_stage)) or "processed"
    )
    industrial._FedotIndustrial__abstract_predict = (
        lambda data, mode: calls.append(("predict", data, mode)) or "predicted"
    )

    result = industrial.predict_proba("raw", predict_mode="probs")

    assert result == "predicted"
    assert industrial.manager.predicted_probs == "predicted"
    assert calls == [
        ("process", "raw", False),
        ("predict", "processed", "labels"),
    ]


def test_shutdown_is_idempotent_after_resources_are_closed():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    industrial.repository_initializer = SimpleNamespace(
        close=lambda: calls.append("repository")
    )
    industrial.manager = SimpleNamespace(
        dask_client=SimpleNamespace(close=lambda: calls.append("client")),
        dask_cluster=SimpleNamespace(close=lambda: calls.append("cluster")),
    )

    industrial.shutdown()
    industrial.shutdown()

    assert calls == ["repository", "client", "cluster", "repository"]
    assert industrial.manager.dask_client is None
    assert industrial.manager.dask_cluster is None


@pytest.mark.parametrize("failed", [
    ("repository",), ("client",), ("cluster",), ("repository", "client", "cluster"),
])
def test_shutdown_attempts_all_resources_and_preserves_the_first_error(failed):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    errors = {name: OSError(f"cannot close {name}") for name in failed}

    def resource(name):
        def close():
            calls.append(name)
            if name in errors:
                raise errors[name]
        return SimpleNamespace(close=close)

    industrial.repository_initializer = resource("repository")
    client, cluster = resource("client"), resource("cluster")
    industrial.manager = SimpleNamespace(dask_client=client, dask_cluster=cluster)

    with pytest.raises(OSError) as caught:
        industrial.shutdown()

    assert caught.value is errors[failed[0]]
    assert calls == ["repository", "client", "cluster"]
    assert industrial.manager.dask_client is (client if "client" in failed else None)
    assert industrial.manager.dask_cluster is (cluster if "cluster" in failed else None)


def test_shutdown_can_retry_a_handle_that_failed_to_close():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []

    def close_client():
        calls.append("client")
        if calls.count("client") == 1:
            raise OSError("temporary close failure")

    industrial.repository_initializer = SimpleNamespace(close=lambda: calls.append("repository"))
    industrial.manager = SimpleNamespace(
        dask_client=SimpleNamespace(close=close_client),
        dask_cluster=SimpleNamespace(close=lambda: calls.append("cluster")),
    )
    client = industrial.manager.dask_client

    industrial.shutdown(raise_cleanup_errors=False)
    assert industrial.manager.dask_client is client
    assert industrial.manager.dask_cluster is None
    industrial.shutdown()

    assert calls == ["repository", "client", "cluster", "repository", "client"]
    assert industrial.manager.dask_client is None


@pytest.mark.parametrize("operation", ["fit", "finetune"])
def test_operation_failure_survives_cleanup_errors_and_all_resources_are_attempted(operation):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    calls = []
    operation_error = ValueError("model preparation failed")

    def fail(*args, **kwargs):
        raise operation_error

    def close_repository():
        calls.append("repository")
        raise OSError("repository close failed")

    def close_client():
        calls.append("client")
        raise RuntimeError("client close failed")

    client = SimpleNamespace(close=close_client)
    industrial.repository_initializer = SimpleNamespace(close=close_repository)
    industrial.manager = SimpleNamespace(
        dask_client=client,
        dask_cluster=SimpleNamespace(close=lambda: calls.append("cluster")),
        condition_check=SimpleNamespace(input_data_is_fedot_type=lambda data: False),
        automl_config=SimpleNamespace(config={"task": "classification"}),
    )
    industrial._process_input_data = fail
    industrial.finetune_service = SimpleNamespace(prepare_payload=fail)

    with pytest.raises(ValueError) as caught:
        getattr(industrial, operation)("raw")

    assert caught.value is operation_error
    assert calls == ["repository", "client", "cluster"]
    assert industrial.manager.dask_client is client
    assert industrial.manager.dask_cluster is None
