import base64
import csv
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from benchmark.industrial.experiments.evolution_parallelism import monitoring
from benchmark.industrial.experiments.evolution_parallelism.monitoring import (
    ProcessTreeMonitor, dask_diagnostics, summarize_process_samples, summarize_task_stream,
)


@given(st.lists(st.tuples(st.integers(0, 5), st.integers(1, 1000),
                          st.integers(1, 30), st.integers(0, 50)), max_size=50))
def test_process_aggregation_conserves_cpu_and_uses_simultaneous_rss(values):
    rows = [dict(sample_id=s, rss_bytes=rss, num_threads=threads,
                 cpu_delta_seconds=cpu, pid=i, create_time=0)
            for i, (s, rss, threads, cpu) in enumerate(values)]
    result = summarize_process_samples(iter(rows))
    sample_ids = {r["sample_id"] for r in rows}
    assert result["cpu_seconds"] == sum(v[3] for v in values)
    assert result["sample_count"] == len(sample_ids)
    assert result["process_count"] == len(rows)
    assert result["peak_rss_sum_bytes"] == max(
        (sum(r["rss_bytes"] for r in rows if r["sample_id"] == s) for s in sample_ids), default=0)
    assert result["peak_thread_count"] == max(
        (sum(r["num_threads"] for r in rows if r["sample_id"] == s) for s in sample_ids), default=0)


@given(st.lists(st.tuples(st.sampled_from(["compute", "transfer", "disk-read", "deserialize"]),
                          st.integers(0, 100)), max_size=50))
def test_task_actions_partition_total_duration_without_counting_payload_as_transfer(actions):
    tasks = [dict(worker="worker-a", nbytes=10**9,
                  startstops=[dict(action=kind, start=10, stop=10 + duration)])
             for kind, duration in actions]
    result = summarize_task_stream(tasks)
    assert result["action_seconds"] == sum(d for _, d in actions)
    assert result["compute_seconds"] == sum(d for k, d in actions if k == "compute")
    assert result["transfer_seconds"] == sum(d for k, d in actions if k == "transfer")
    assert result["task_count"] == len(tasks)
    assert result["empty_tasks"] == (not tasks)
    assert summarize_task_stream(reversed(tasks)) == result


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_task_duration_is_not_silently_clamped(duration):
    with pytest.raises(ValueError, match="duration"):
        summarize_task_stream([dict(worker="a", startstops=[
            dict(action="compute", start=0, stop=duration)])])


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_process_interval(interval, tmp_path):
    with pytest.raises(ValueError, match="interval"):
        ProcessTreeMonitor(tmp_path, interval)


@pytest.mark.parametrize("fail", [False, True])
def test_process_monitor_joins_and_exports_on_exit(tmp_path, fail):
    monitor = ProcessTreeMonitor(tmp_path, interval=0.01)
    original = LookupError("search failed")
    try:
        with monitor:
            time.sleep(0.035)
            if fail:
                raise original
    except LookupError as error:
        assert fail and error is original
    assert monitor._thread is not None and not monitor._thread.is_alive()
    assert monitor.summary["sample_count"] >= 2
    assert monitor.summary["peak_rss_sum_bytes"] > 0
    assert monitor.summary["errors"] == []
    assert "not unique" in monitor.summary["rss_semantics"]
    with (tmp_path / "process_samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and str(os.getpid()) in {r["pid"] for r in rows}
    assert all(float(r["cpu_delta_seconds"]) >= 0 for r in rows)
    assert all(int(r["num_threads"]) > 0 for r in rows)
    with pytest.raises(RuntimeError, match="single-use"):
        monitor.__enter__()


def test_process_monitor_observes_child_without_stopping_it(tmp_path):
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as child:
        try:
            with ProcessTreeMonitor(tmp_path, interval=0.01) as monitor:
                time.sleep(0.03)
            assert child.poll() is None
            assert child.pid in {r["pid"] for r in monitor.samples}
            assert monitor.summary["peak_process_count"] >= 2
        finally:
            child.terminate()
            child.wait(timeout=10)


def test_sampling_failure_is_reported_and_thread_still_joins(tmp_path, monkeypatch):
    monitor = ProcessTreeMonitor(tmp_path, interval=0.01)
    sampled = threading.Event()
    with pytest.raises(RuntimeError, match="sampling broke"):
        with monitor:
            def broken_sample():
                sampled.set()
                raise OSError("sampling broke")

            monkeypatch.setattr(monitor, "_sample", broken_sample)
            assert sampled.wait(5)
    assert not monitor._thread.is_alive()
    assert (tmp_path / "process_samples.csv").is_file()


def test_process_write_failure_preserves_original_exception(tmp_path, monkeypatch):
    original = ValueError("original")
    monitor = ProcessTreeMonitor(tmp_path)

    def fail_write(*args):
        raise OSError("disk failed")

    monkeypatch.setattr(monitoring, "_write_csv", fail_write)
    with pytest.warns(RuntimeWarning, match="disk failed"):
        with pytest.raises(ValueError) as caught:
            with monitor:
                raise original
    assert caught.value is original
    assert not monitor._thread.is_alive()


def test_process_identity_baselines_and_pid_reuse(tmp_path):
    monitor = ProcessTreeMonitor(tmp_path)
    monitor._start = time.monotonic()
    monitor._start_timestamp = 100

    class Process:
        pid = 123
        created = 90
        cpu = 20

        def oneshot(self):
            from contextlib import nullcontext
            return nullcontext()

        def children(self, recursive):
            assert recursive
            return []

        def create_time(self):
            return self.created

        def cpu_times(self):
            return SimpleNamespace(user=self.cpu, system=0)

        def memory_info(self):
            return SimpleNamespace(rss=100)

        def ppid(self):
            return 1

        def num_threads(self):
            return 2

    monitor._owner = process = Process()
    monitor._sample()
    process.cpu = 23
    monitor._sample()
    process.created, process.cpu = 101, 2
    monitor._sample()
    assert [r["cpu_delta_seconds"] for r in monitor.samples] == [0, 3, 2]
    assert monitor.summary["process_count"] == 2


@pytest.fixture(scope="module")
def dask_client():
    distributed = pytest.importorskip("distributed")
    pytest.importorskip("bokeh")
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                                  dashboard_address=None, memory_limit=0) as cluster:
        with distributed.Client(cluster, set_as_default=False) as client:
            yield client


def _tiny_task(value):
    time.sleep(0.025)
    return value + 1


def test_real_dask_artifacts_explicit_client_and_repeated_batches(dask_client, tmp_path, monkeypatch):
    from distributed import get_client
    from distributed.diagnostics import MemorySampler

    actual_sample = MemorySampler.sample
    seen = []

    def sample(self, label, *, client, **kwargs):
        seen.append(client)
        return actual_sample(self, label, client=client, **kwargs)

    monkeypatch.setattr(MemorySampler, "sample", sample)
    for batch in range(2):
        output = tmp_path / f"batch-{batch}"
        with dask_diagnostics(dask_client, output, label="population") as record:
            with pytest.raises(ValueError, match="No global client"):
                get_client()
            futures = dask_client.map(_tiny_task, range(4), pure=False)
            assert dask_client.gather(futures) == [1, 2, 3, 4]
            # Worker heartbeat updates MemorySampler's scheduler-side RSS.
            time.sleep(0.25)
        with pytest.raises(ValueError, match="No global client"):
            get_client()
        assert record.summary["errors"] == []
        assert record.summary["task_count"] == 4
        assert not record.summary["empty_tasks"]
        assert record.summary["compute_seconds"] > 0
        assert record.summary["memory_sample_count"] >= 1
        assert record.summary["profile_status"] in {"empty", "collected"}
        assert (output / "worker_profile.html").exists() == (
            record.summary["profile_status"] == "collected")

        def decode_bytes(value):
            if set(value) == {"__bytes_base64__"}:
                return base64.b64decode(value["__bytes_base64__"])
            return value

        assert json.loads((output / "task_stream.json").read_text(),
                          object_hook=decode_bytes) == record.tasks
        for filename in ("task_stream.html", "performance_report.html"):
            assert "<html" in (output / filename).read_text(encoding="utf-8").lower()
        with (output / "memory_samples.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(record.memory_samples)
        assert {r["label"] for r in rows} == {"population"}
        assert dask_client.status == "running"
        assert dask_client.cluster.status.value == "running"
        with pytest.raises(FileExistsError):
            with dask_diagnostics(dask_client, output):
                pytest.fail("must not overwrite diagnostics")
    assert seen == [dask_client, dask_client]


def test_real_dask_empty_stream_and_exception(dask_client, tmp_path):
    original = LookupError("search body failed")
    with pytest.raises(LookupError) as caught:
        with dask_diagnostics(dask_client, tmp_path, label="empty") as record:
            raise original
    assert caught.value is original
    assert record.summary["empty_tasks"]
    assert record.summary["task_count"] == 0
    assert record.summary["errors"] == []
    assert json.loads((tmp_path / "task_stream.json").read_text()) == []
    assert (tmp_path / "task_stream.html").is_file()
    assert (tmp_path / "performance_report.html").is_file()
    assert dask_client.status == "running"


@pytest.mark.parametrize("body_fails", [False, True])
def test_report_failure_exports_other_collectors_and_preserves_body(
        dask_client, tmp_path, monkeypatch, body_fails):
    import distributed

    @contextmanager
    def broken_report(**kwargs):
        try:
            yield
        finally:
            raise OSError("report unavailable")

    monkeypatch.setattr(distributed, "performance_report", broken_report)
    if body_fails:
        with pytest.warns(RuntimeWarning, match="report"):
            with pytest.raises(LookupError, match="original"):
                with dask_diagnostics(dask_client, tmp_path) as record:
                    raise LookupError("original")
    else:
        with pytest.raises(RuntimeError, match="report unavailable"):
            with dask_diagnostics(dask_client, tmp_path) as record:
                pass
    assert record.summary["errors"]
    assert (tmp_path / "task_stream.json").is_file()
    assert (tmp_path / "memory_samples.csv").is_file()
    assert (tmp_path / "task_stream.html").is_file()
    assert dask_client.status == "running"


def test_async_client_is_rejected_before_writes(tmp_path):
    pytest.importorskip("distributed")
    target = tmp_path / "unused"
    with pytest.raises(ValueError, match="synchronous"):
        with dask_diagnostics(SimpleNamespace(asynchronous=True), target):
            pytest.fail("async client must not enter")
    assert not target.exists()


def test_plot_failure_preserves_raw_tasks_and_memory(dask_client, tmp_path, monkeypatch):
    from distributed.dashboard.components import scheduler

    def broken_plot(**kwargs):
        raise OSError("plot unavailable")

    monkeypatch.setattr(scheduler, "task_stream_figure", broken_plot)
    with pytest.raises(RuntimeError, match="plot unavailable"):
        with dask_diagnostics(dask_client, tmp_path) as record:
            future = dask_client.submit(_tiny_task, 5, pure=False)
            assert future.result() == 6
    assert record.summary["task_count"] == 1
    assert len(json.loads((tmp_path / "task_stream.json").read_text())) == 1
    assert (tmp_path / "memory_samples.csv").is_file()
    assert dask_client.status == "running"


def test_current_client_is_only_scoped_to_report(dask_client, tmp_path, monkeypatch):
    import distributed

    actual_report = distributed.performance_report
    calls = []

    @contextmanager
    def checked_report(**kwargs):
        assert distributed.get_client() is dask_client
        calls.append("enter")
        try:
            with actual_report(**kwargs):
                yield
        finally:
            assert distributed.get_client() is dask_client
            calls.append("exit")

    monkeypatch.setattr(distributed, "performance_report", checked_report)
    with distributed.Client(dask_client.cluster, set_as_default=False) as ambient:
        with ambient.as_current():
            with dask_diagnostics(dask_client, tmp_path):
                assert distributed.get_client() is ambient
            assert distributed.get_client() is ambient
    assert calls == ["enter", "exit"]


def test_optional_profile_failure_is_explicit(dask_client, tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("worker profiling disabled")

    monkeypatch.setattr(dask_client, "profile", unavailable)
    with pytest.warns(RuntimeWarning, match="worker profiling disabled"):
        with dask_diagnostics(dask_client, tmp_path) as record:
            pass
    assert record.summary["errors"] == []
    assert record.summary["profile_status"] == "unavailable"
    assert "worker profiling disabled" in record.summary["profile_error"]
    assert not (tmp_path / "worker_profile.html").exists()


@pytest.mark.parametrize("label", ["", None, 0])
def test_empty_memory_label_is_rejected(tmp_path, label):
    pytest.importorskip("distributed")
    with pytest.raises(ValueError, match="label"):
        with dask_diagnostics(SimpleNamespace(asynchronous=False), tmp_path, label):
            pytest.fail("MemorySampler substitutes its own key for empty labels")
