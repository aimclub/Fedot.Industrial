"""Owned Dask processes: admission, exact terminal results, cancellation and recovery."""

from dataclasses import replace
import json
import os
from pathlib import Path
import random
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
import cloudpickle

from fedot_ind.core.optimizer.evaluation.contracts import (
    EvaluationPlan, EvaluationTask, ExecutionError, ExecutionResult, ExecutionStatus,
)
from fedot_ind.core.optimizer.evaluation.dask_config import DaskEvaluationPlan, DaskExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_executor import (
    DaskCleanupError, DaskEvaluationSession, _catalog_snapshot, execute_dask_plan, retry_worker_failure,
)
from fedot_ind.core.optimizer.evaluation.executors import CancellationToken, execute_plan


def _draw(delay):
    sleep(delay)
    return random.random(), float(np.random.normal())


def _mixed(payload):
    kind, path = payload
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")
    if kind == "fail":
        raise ValueError("deliberate model failure")
    return kind


def _crash_once(payload):
    kind, path = payload
    path = Path(path)
    previous = path.read_text() if path.exists() else ""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")
    if kind == "crash" and not previous:
        os._exit(19)
    return kind


def _long_fit(path):
    Path(path).write_text(str(os.getpid()), encoding="utf-8")
    sleep(60)
    return "late success"


def _sleep_or_value(payload):
    if isinstance(payload, str):
        return _long_fit(payload)
    return payload


def _unpickleable(_):
    class Broken:
        def __reduce__(self):
            raise TypeError("cannot serialize result")
    return Broken()


class _UnserializableInput:
    def __reduce__(self):
        raise TypeError("cannot serialize input")


def _identity(value):
    return value


def _persist_then_exit(task, payload, evaluate, digest, result_path):
    from fedot_ind.core.optimizer.evaluation.dask_bootstrap import evaluate_on_worker
    evaluate_on_worker(task, payload, evaluate, digest, result_path)
    os._exit(21)


def test_dask_draws_match_sequential_and_plan_order():
    tasks = (EvaluationTask("slow", 42), EvaluationTask("fast", 123))
    sequential = execute_plan(EvaluationPlan(tasks), [.2, 0.], _draw)
    attempts = []
    actual = execute_dask_plan(DaskEvaluationPlan(tasks, 2), [.2, 0.], _draw, on_attempt=attempts.append)
    assert [result.candidate_id for result in actual] == ["slow", "fast"]
    assert all(result.status is ExecutionStatus.SUCCEEDED for result in actual)
    assert [result.value for result in actual] == [result.value for result in sequential]
    assert {item.candidate_id for item in attempts} == {"slow", "fast"}
    assert all(item.number == 1 for item in attempts)


def test_model_failure_does_not_retry_or_destroy_the_batch(tmp_path):
    paths = [tmp_path / "fail.txt", tmp_path / "good.txt"]
    result = execute_dask_plan(
        DaskEvaluationPlan((EvaluationTask("bad"), EvaluationTask("good")), 2),
        [("fail", str(paths[0])), ("good", str(paths[1]))], _mixed,
        config=DaskExecutionConfig(retries=3))
    assert result[0].status is ExecutionStatus.FAILED
    assert result[0].error.error_type == "ValueError"
    assert result[1].status is ExecutionStatus.SUCCEEDED
    assert all(len(path.read_text().splitlines()) == 1 for path in paths)


@pytest.mark.parametrize("retries", [0, 1])
def test_worker_crash_has_only_explicit_bounded_retries(tmp_path, retries):
    paths = [tmp_path / "crash.txt", tmp_path / "good.txt"]
    attempts = []
    result = execute_dask_plan(
        DaskEvaluationPlan((EvaluationTask("crash"), EvaluationTask("good")), 2),
        [("crash", str(paths[0])), ("good", str(paths[1]))], _crash_once,
        config=DaskExecutionConfig(retries=retries), on_attempt=attempts.append)
    assert result[1].status is ExecutionStatus.SUCCEEDED
    assert result[0].status is (ExecutionStatus.SUCCEEDED if retries else ExecutionStatus.WORKER_FAILED)
    assert len(paths[0].read_text().splitlines()) == retries + 1
    assert len(paths[1].read_text().splitlines()) == 1
    assert len([item for item in attempts if item.candidate_id == "crash"]) == retries + 1


def test_candidate_timeout_kills_owned_worker_and_next_candidate_runs(tmp_path):
    marker = tmp_path / "slow.txt"
    results = execute_dask_plan(
        DaskEvaluationPlan((EvaluationTask("slow"), EvaluationTask("good")), 1, 2.),
        [str(marker), 17], _sleep_or_value)
    assert results[0].status is ExecutionStatus.TIMED_OUT
    assert results[1].status is ExecutionStatus.SUCCEEDED
    assert results[1].value == 17
    assert marker.exists()
    assert not psutil.pid_exists(int(marker.read_text()))


def test_cancellation_interrupts_running_fit_and_closes_process(tmp_path):
    marker = tmp_path / "cancel.txt"
    token = CancellationToken()

    def cancel_after_start():
        deadline = monotonic() + 30
        while not marker.exists() and monotonic() < deadline:
            sleep(.02)
        token.cancel()

    watcher = Thread(target=cancel_after_start)
    watcher.start()
    try:
        result, = execute_dask_plan(DaskEvaluationPlan((EvaluationTask("slow"),)), [str(marker)], _long_fit,
                                    cancellation=token)
    finally:
        token.cancel()
        watcher.join(timeout=31)
    assert marker.exists()
    assert result.status is ExecutionStatus.CANCELLED
    assert not psutil.pid_exists(int(marker.read_text()))


def test_precancelled_and_empty_plans_never_start_dask(monkeypatch):
    import fedot_ind.core.optimizer.evaluation.dask_executor as module

    def forbidden(*args, **kwargs):
        pytest.fail("Precancelled/empty execution started a cluster")

    monkeypatch.setattr(module, "DaskEvaluationSession", forbidden)
    token = CancellationToken()
    token.cancel()
    result, = execute_dask_plan(DaskEvaluationPlan((EvaluationTask("a"),)), [0], _draw, cancellation=token)
    assert result.status is ExecutionStatus.CANCELLED
    assert execute_dask_plan(DaskEvaluationPlan(()), [], _draw) == ()


def test_session_manifest_resources_and_owned_cleanup():
    from distributed import default_client
    from fedot_ind.core.optimizer.evaluation.dask_bootstrap import worker_resources

    try:
        before = default_client()
    except ValueError:
        before = None
    with DaskEvaluationSession(1, DaskExecutionConfig()) as session:
        reports = session.client.run(worker_resources)
        assert all(row["digest"] == session.digest and row["threads"] == 1 and row["memory_limit"] > 0
                   for row in reports.values())
        pids = set(session.worker_pids)
        assert pids and os.getpid() not in pids
        if before is None:
            with pytest.raises(ValueError):
                default_client()
        else:
            assert default_client() is before
    assert session.client is session.cluster is None
    assert not any(psutil.pid_exists(pid) for pid in pids)
    session.close()


def test_bootstrap_rejects_mismatched_manifest_without_touching_registry():
    from fedot_ind.core.optimizer.evaluation.dask_bootstrap import IndustrialWorkerBootstrap
    digest, payload = _catalog_snapshot()
    value = json.loads(payload)
    value["manifest"]["version"] = "different"
    worker = SimpleNamespace(state=SimpleNamespace(nthreads=1), memory_manager=SimpleNamespace(memory_limit=1024))
    with pytest.raises(ValueError, match="catalog differs"):
        IndustrialWorkerBootstrap(json.dumps(value)).setup(worker)
    assert not hasattr(worker, "_industrial_evaluation_manifest")
    IndustrialWorkerBootstrap(payload).setup(worker)
    assert worker._industrial_evaluation_manifest == digest


def test_private_session_never_closes_an_ambient_client():
    from distributed import Client, LocalCluster, default_client
    with LocalCluster(n_workers=0, host="127.0.0.1", dashboard_address=None,
                      scheduler_kwargs={"dashboard_address": "127.0.0.1:0"}) as cluster:
        with Client(cluster, set_as_default=True) as ambient:
            with DaskEvaluationSession(1, DaskExecutionConfig()):
                assert default_client() is ambient
            assert default_client() is ambient
            assert ambient.status == "running"


def test_startup_failure_closes_partially_created_resources(monkeypatch):
    import distributed
    from unittest.mock import MagicMock
    cluster, client = MagicMock(), MagicMock()
    client.register_plugin.side_effect = ValueError("invalid bootstrap")
    monkeypatch.setattr(distributed, "LocalCluster", lambda **kwargs: cluster)
    monkeypatch.setattr(distributed, "Client", lambda *args, **kwargs: client)
    session = DaskEvaluationSession(1, DaskExecutionConfig())
    with pytest.raises(ValueError, match="invalid bootstrap"):
        with session:
            pytest.fail("Entered a session after a startup failure")
    client.close.assert_called_once()
    cluster.close.assert_called_once()
    assert session.client is session.cluster is None


def test_unconfirmed_cleanup_is_not_silently_returned_as_batch_success(monkeypatch):
    import fedot_ind.core.optimizer.evaluation.dask_executor as module

    class BrokenCleanup:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise DaskCleanupError("worker did not stop")

    monkeypatch.setattr(module, "DaskEvaluationSession", BrokenCleanup)
    monkeypatch.setattr(module, "_run", lambda *args: (ExecutionResult("a", ExecutionStatus.SUCCEEDED, 0., value=1),))
    with pytest.raises(DaskCleanupError):
        execute_dask_plan(DaskEvaluationPlan((EvaluationTask("a"),)), [0], _draw)


def test_total_deadline_includes_cluster_startup(monkeypatch):
    import fedot_ind.core.optimizer.evaluation.dask_executor as module

    class SlowStartup:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            sleep(.03)
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "DaskEvaluationSession", SlowStartup)
    result, = execute_dask_plan(DaskEvaluationPlan((EvaluationTask("a"),), total_timeout_seconds=.01), [0], _draw)
    assert result.status is ExecutionStatus.TIMED_OUT


@pytest.mark.parametrize("worker_cleanup_fails", [False, True])
def test_result_directory_failure_cannot_hide_fatal_cleanup(tmp_path, monkeypatch, worker_cleanup_fails):
    import fedot_ind.core.optimizer.evaluation.dask_executor as module

    class BrokenDirectory:
        name = str(tmp_path)

        def cleanup(self):
            raise PermissionError("directory still in use")

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if worker_cleanup_fails:
                raise DaskCleanupError("worker did not stop")

    monkeypatch.setattr(module, "TemporaryDirectory", lambda **kwargs: BrokenDirectory())
    monkeypatch.setattr(module, "DaskEvaluationSession", Session)
    monkeypatch.setattr(module, "_run", lambda *args: (ExecutionResult("a", ExecutionStatus.SUCCEEDED, 0., value=1),))
    with pytest.raises(DaskCleanupError):
        execute_dask_plan(DaskEvaluationPlan((EvaluationTask("a"),)), [0], _draw)


def test_unserializable_result_is_structured_failure():
    result, = execute_dask_plan(DaskEvaluationPlan((EvaluationTask("a"),)), [0], _unpickleable)
    assert result.status is ExecutionStatus.WORKER_FAILED


def test_unserializable_input_does_not_cancel_healthy_neighbors():
    results = execute_dask_plan(
        DaskEvaluationPlan(tuple(EvaluationTask(str(i)) for i in range(3)), 2),
        [11, _UnserializableInput(), 17], _identity)
    assert [result.status for result in results] == [
        ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.SUCCEEDED,
    ]
    assert [results[0].value, results[2].value] == [11, 17]
    assert "serialize" in results[1].error.message.lower()


def test_completed_result_survives_worker_death_before_receipt(tmp_path, monkeypatch):
    import fedot_ind.core.optimizer.evaluation.dask_bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "evaluate_on_worker", _persist_then_exit)
    marker = tmp_path / "once.txt"
    result, = execute_dask_plan(
        DaskEvaluationPlan((EvaluationTask("a"),), candidate_timeout_seconds=10., total_timeout_seconds=30.),
        [("good", str(marker))], _mixed)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.value == "good"
    assert len(marker.read_text().splitlines()) == 1


def test_recomputing_a_receipt_does_not_repeat_the_model(tmp_path):
    from fedot_ind.core.optimizer.evaluation.dask_bootstrap import evaluate_on_worker
    result_path = tmp_path / "result.pkl"
    marker = tmp_path / "once.txt"
    with DaskEvaluationSession(1, DaskExecutionConfig()) as session:
        for _ in range(2):
            receipt = session.client.submit(
                evaluate_on_worker, EvaluationTask("a"), ("good", str(marker)), _mixed,
                session.digest, str(result_path), pure=False).result(timeout=20)
            assert receipt == str(result_path)
    assert len(marker.read_text().splitlines()) == 1
    assert cloudpickle.loads(result_path.read_bytes()).value == "good"


def test_receive_never_gathers_a_lost_computed_value(tmp_path):
    from unittest.mock import MagicMock
    from fedot_ind.core.optimizer.evaluation.dask_executor import _receive
    path = tmp_path / "result.pkl"
    path.write_bytes(cloudpickle.dumps(ExecutionResult("a", ExecutionStatus.SUCCEEDED, 0., value=13)))
    future = MagicMock(status="pending")
    future.result.side_effect = AssertionError("Would block waiting for lost Dask data")
    result = _receive(SimpleNamespace(task=EvaluationTask("a"), future=future,
                                      result_path=path, started=monotonic()))
    assert result.value == 13
    future.result.assert_not_called()
    future.exception.assert_not_called()


@pytest.mark.parametrize("status", list(ExecutionStatus))
def test_retry_rule_never_retries_model_failures_or_exceeds_limit(status):
    result = ExecutionResult("a", status, 0., error=None if status is ExecutionStatus.SUCCEEDED
                             else ExecutionError("KilledWorker", "lost"))
    assert retry_worker_failure(result, 0, 1) is (status is ExecutionStatus.WORKER_FAILED)
    assert not retry_worker_failure(result, 1, 1)
    if result.error is not None:
        assert not retry_worker_failure(replace(result, error=ExecutionError("ValueError", "model failed")), 0, 1)
