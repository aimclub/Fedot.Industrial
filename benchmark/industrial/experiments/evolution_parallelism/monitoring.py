"""Diagnostics for bounded search demos; no ownership of clients or workers.

Pass an ignored artifact directory (for example under ``.local``). Give every
population batch its own directory, even when each batch uses a new cluster.
RSS values are sums of process RSS, NOT unique physical memory or PSS. Polling
cannot see processes that start and exit between samples. Dask's task-stream
buffer is bounded by scheduler configuration; these contexts do not change it.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import sys
import threading
import time
import warnings
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import psutil


PROCESS_FIELDS = (
    "sample_id", "timestamp", "elapsed_seconds", "pid", "create_time", "ppid",
    "is_owner", "rss_bytes", "cpu_user_seconds", "cpu_system_seconds",
    "cpu_delta_seconds", "num_threads",
)
RSS_DESCRIPTION = "sum of process RSS; not unique physical memory or PSS"


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    # distributed 2024.4.2 includes a pickled type in task-stream records.
    # Preserve its bytes without executing or silently stringifying the pickle.
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Cannot encode {type(value).__name__} as task-stream JSON")


def summarize_process_samples(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-process rows by sampling instant, not by PID lifetime."""
    samples = {}
    identities = set()
    cpu_seconds = 0.0
    for row in rows:
        sample = samples.setdefault(row["sample_id"], [0, 0, 0])
        sample[0] += row["rss_bytes"]
        sample[1] += row["num_threads"]
        sample[2] += 1
        identities.add((row["pid"], row["create_time"]))
        cpu_seconds += row["cpu_delta_seconds"]
    return {
        "sample_count": len(samples),
        "process_count": len(identities),
        "peak_rss_sum_bytes": max((s[0] for s in samples.values()), default=0),
        "peak_thread_count": max((s[1] for s in samples.values()), default=0),
        "peak_process_count": max((s[2] for s in samples.values()), default=0),
        "cpu_seconds": cpu_seconds,
        "rss_semantics": RSS_DESCRIPTION,
    }


class ProcessTreeMonitor:
    """Sample this process and recursive children; never stop any process.

    ``root`` is the output directory, not a PID. ``samples`` contains one row
    per observed process per instant. CPU deltas use (PID, creation time) to
    resist PID reuse; the first observation is a baseline for existing
    processes, while newly spawned children contribute their lifetime CPU.
    The context is single-use. ``summary`` is final after exiting it.
    """

    def __init__(self, root: Path, interval: float = 0.2):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.root = Path(root)
        self.interval = interval
        self.samples = []
        self.errors = []
        self.skipped_observations = 0
        self._stop = threading.Event()
        self._thread = None
        self._previous_cpu = {}
        self._sample_id = 0
        self._start = None
        self._elapsed = 0.0

    @property
    def summary(self):
        return {
            **summarize_process_samples(self.samples),
            "elapsed_seconds": self._elapsed,
            "skipped_observations": self.skipped_observations,
            "errors": list(self.errors),
        }

    def _sample(self):
        timestamp = time.time()
        elapsed = time.monotonic() - self._start
        processes = [self._owner, *self._owner.children(recursive=True)]
        seen = set()
        for process in processes:
            try:
                with process.oneshot():
                    created = process.create_time()
                    identity = (process.pid, created)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    cpu = process.cpu_times()
                    cpu_total = cpu.user + cpu.system
                    baseline = 0.0 if created >= self._start_timestamp else cpu_total
                    previous = self._previous_cpu.get(identity, baseline)
                    row = dict(
                        sample_id=self._sample_id, timestamp=timestamp,
                        elapsed_seconds=elapsed, pid=process.pid,
                        create_time=created, ppid=process.ppid(),
                        is_owner=process.pid == self._owner.pid,
                        rss_bytes=process.memory_info().rss,
                        cpu_user_seconds=cpu.user, cpu_system_seconds=cpu.system,
                        cpu_delta_seconds=max(0.0, cpu_total - previous),
                        num_threads=process.num_threads(),
                    )
                self.samples.append(row)
                self._previous_cpu[identity] = cpu_total
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.skipped_observations += 1
        self._sample_id += 1

    def _run(self):
        try:
            while not self._stop.wait(self.interval):
                self._sample()
        except Exception as error:
            self.errors.append(f"sampling: {type(error).__name__}: {error}")
            self._stop.set()

    def __enter__(self):
        if self._start is not None:
            raise RuntimeError("ProcessTreeMonitor is single-use")
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "process_samples.csv").exists():
            raise FileExistsError(self.root / "process_samples.csv")
        self._owner = psutil.Process(os.getpid())
        self._start_timestamp = time.time()
        self._start = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="process-tree-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        try:
            self._sample()
        except Exception as error:
            self.errors.append(f"final sampling: {type(error).__name__}: {error}")
        self._elapsed = time.monotonic() - self._start
        try:
            _write_csv(self.root / "process_samples.csv", PROCESS_FIELDS, self.samples)
        except Exception as error:
            self.errors.append(f"process_samples.csv: {type(error).__name__}: {error}")
        if self.errors:
            message = "; ".join(self.errors)
            if exc_type is None:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False


def summarize_task_stream(tasks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum task action durations (parallel intervals can overlap).

    ``task_count`` counts execution records, including retries, not unique keys.
    Transfer is only the explicit ``transfer`` action, not task payload bytes.
    """
    durations = {}
    workers = set()
    count = 0
    for task in tasks:
        count += 1
        workers.add(task["worker"])
        for action in task.get("startstops", ()):
            duration = action["stop"] - action["start"]
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("task action duration must be finite and nonnegative")
            name = action["action"]
            durations[name] = durations.get(name, 0.0) + duration
    return {
        "task_count": count, "empty_tasks": count == 0,
        "worker_count": len(workers),
        "compute_seconds": durations.get("compute", 0.0),
        "transfer_seconds": durations.get("transfer", 0.0),
        "action_seconds": sum(durations.values()),
        "action_seconds_by_kind": durations,
    }


@dataclass
class DaskDiagnosticsRecord:
    output_dir: Path
    label: str
    tasks: list = field(default_factory=list)
    memory_samples: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    profile_status: str = "not_collected"
    profile_error: str | None = None

    @property
    def summary(self):
        return {
            **summarize_task_stream(self.tasks),
            "label": self.label, "elapsed_seconds": self.elapsed_seconds,
            "memory_sample_count": len(self.memory_samples),
            "peak_worker_rss_sum_bytes": max((s[1] for s in self.memory_samples), default=0),
            "rss_semantics": RSS_DESCRIPTION,
            "profile_status": self.profile_status, "profile_error": self.profile_error,
            "errors": list(self.errors),
        }


@contextmanager
def _diagnostic_context(context, record, name, current_client=None):
    """Finish every collector and retain the search's original exception."""
    with current_client.as_current() if current_client is not None else nullcontext():
        value = context.__enter__()
    try:
        yield value
    finally:
        try:
            with current_client.as_current() if current_client is not None else nullcontext():
                context.__exit__(*sys.exc_info())
        except Exception as error:
            record.errors.append(f"{name}: {type(error).__name__}: {error}")


@contextmanager
def dask_diagnostics(client, output_dir: Path, label: str = "search"):
    """Collect diagnostics from an existing synchronous distributed client.

    Does not create/close clients or clusters or change scheduler configuration.
    Writes the four required artifacts directly into ``output_dir``; callers
    must choose distinct nested directories for batches. Existing artifacts
    are rejected to prevent accidental overwrite. An empty task stream is a
    real empty JSON list/plot with ``summary['empty_tasks'] == True``.
    Required collector failures raise RuntimeError after export; on a search
    exception they are recorded and warned without replacing that exception.
    A worker profile is optional and written only with actual stack samples.
    Binary JSON values use ``{"__bytes_base64__": "..."}`` (no unpickling).
    Only performance_report entry/exit uses ``client.as_current()``; the search
    body retains its caller's current-client context.
    """
    from distributed import get_task_stream, performance_report
    from distributed.diagnostics import MemorySampler

    if client.asynchronous:
        raise ValueError("dask_diagnostics requires a synchronous client")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a nonempty string")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("performance_report.html", "task_stream.html", "task_stream.json",
             "memory_samples.csv", "worker_profile.html")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    record = DaskDiagnosticsRecord(output_dir, label)
    sampler = MemorySampler()
    tasks = None
    started = time.time()
    monotonic_start = time.monotonic()
    try:
        with ExitStack() as stack:
            stack.enter_context(_diagnostic_context(
                performance_report(filename=str(output_dir / names[0])), record, names[0],
                current_client=client))
            tasks = stack.enter_context(_diagnostic_context(
                get_task_stream(client=client), record, "task stream collection"))
            stack.enter_context(_diagnostic_context(
                sampler.sample(label, client=client, measure="process", interval=0.2),
                record, names[3]))
            yield record
    finally:
        body_failed = sys.exc_info()[0] is not None
        stopped = time.time()
        record.elapsed_seconds = time.monotonic() - monotonic_start
        record.tasks = list(tasks.data) if tasks is not None else []
        record.memory_samples = list(sampler.samples.get(label, ()))
        try:
            # Match Client._get_task_stream plotting in distributed 2024.4.2,
            # but keep raw collection independent of rendering failures.
            from bokeh.plotting import save
            from bokeh.resources import CDN
            from distributed.dashboard.components.scheduler import task_stream_figure
            from distributed.diagnostics.task_stream import rectangles

            source, figure = task_stream_figure(sizing_mode="stretch_both")
            source.data.update(rectangles(record.tasks))
            save(figure, filename=str(output_dir / names[1]), resources=CDN,
                 title="Dask Task Stream")
        except Exception as error:
            record.errors.append(f"task_stream.html: {type(error).__name__}: {error}")
        try:
            with (output_dir / names[2]).open("w", encoding="utf-8") as stream:
                json.dump(record.tasks, stream, indent=2, allow_nan=False, default=_json_default)
        except Exception as error:
            record.errors.append(f"task_stream.json: {type(error).__name__}: {error}")
        try:
            _write_csv(output_dir / names[3], ("label", "timestamp", "rss_sum_bytes"),
                       (dict(label=label, timestamp=t, rss_sum_bytes=n)
                        for t, n in record.memory_samples))
        except Exception as error:
            record.errors.append(f"memory_samples.csv: {type(error).__name__}: {error}")
        for name in names[:4]:
            if not (output_dir / name).is_file():
                record.errors.append(f"missing required artifact: {name}")
        try:
            state = client.profile(start=started, stop=stopped)
            if state.get("count", 0) > 0 and state.get("children"):
                from bokeh.plotting import save
                from bokeh.resources import INLINE
                from distributed import profile

                figure, _ = profile.plot_figure(profile.plot_data(state))
                save(figure, filename=str(output_dir / names[4]), resources=INLINE,
                     title="Dask Worker Profile")
                record.profile_status = "collected"
            else:
                record.profile_status = "empty"
        except Exception as error:
            record.profile_status = "unavailable"
            record.profile_error = f"{type(error).__name__}: {error}"
            warnings.warn(f"worker profile: {record.profile_error}", RuntimeWarning, stacklevel=2)
        if record.errors:
            message = "; ".join(record.errors)
            if not body_failed:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
