"""Explicit owned Dask execution; no global/default client and no delayed graph.

One candidate is pinned to one single-thread worker. A deadline kills that
owned worker, not just its Future. Scheduler loss retries are disabled; only
this coordinator may request a bounded infrastructure retry.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from uuid import uuid4

import cloudpickle

from .contracts import ExecutionError, ExecutionResult, ExecutionStatus
from .dask_config import DaskEvaluationPlan, DaskExecutionConfig
from .executors import CancellationToken, _stop_reason


@dataclass(frozen=True)
class DaskAttempt:
    candidate_id: str
    number: int
    worker: str
    status: ExecutionStatus


@dataclass(frozen=True)
class _Flight:
    task: object
    payload: object
    attempt: int
    future: object
    worker: str
    started: float
    result_path: Path


class DaskCleanupError(RuntimeError):
    """Resource ownership cannot be confirmed; the caller must stop the job."""


@contextmanager
def _result_directory():
    directory = TemporaryDirectory(prefix="industrial-dask-")
    try:
        yield Path(directory.name)
    finally:
        try:
            directory.cleanup()
        except Exception as error:
            # Never mask an inner fatal resource error as a recoverable batch
            # failure when outer temporary-file cleanup also fails.
            raise DaskCleanupError("Could not confirm Dask result directory cleanup") from error


def retry_worker_failure(result, attempt, limit):
    """Only explicit infrastructure loss is retryable, never a model exception."""
    return (result.status is ExecutionStatus.WORKER_FAILED and attempt < limit
            and result.error.error_type in {"KilledWorker", "CommClosedError", "WorkerLost"})


def _failed(task, status, started, error_type, message):
    return ExecutionResult(task.candidate_id, status, max(0., monotonic() - started),
                           error=ExecutionError(error_type, message))


def _catalog_snapshot():
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog

    catalog = load_industrial_extension_catalog()
    payload = catalog.to_dict()
    payload.pop("digest")
    return catalog.digest, json.dumps(payload, sort_keys=True, allow_nan=False)


class DaskEvaluationSession:
    """Own only the private LocalCluster and Client created by this instance."""

    def __init__(self, max_workers, config, *, startup_timeout=None, cancellation=None):
        self.max_workers, self.config = max_workers, config
        self.startup_timeout = config.startup_timeout_seconds if startup_timeout is None else startup_timeout
        self.client = self.cluster = None
        self.worker_pids = set()
        self.digest = None
        self._verified = set()
        self.worker_addresses = set()
        self.cancellation = CancellationToken() if cancellation is None else cancellation

    def __enter__(self):
        from distributed import Client, LocalCluster
        from distributed.system import MEMORY_LIMIT
        from .dask_bootstrap import IndustrialWorkerBootstrap

        try:
            self.digest, catalog_json = _catalog_snapshot()
            memory_limit = (max(1, MEMORY_LIMIT // self.max_workers) if self.config.memory_limit == "auto"
                            else self.config.memory_limit)
            started = monotonic()
            # Start empty so a bounded wait owns every partially started resource.
            self.cluster = LocalCluster(
                n_workers=0, threads_per_worker=1, processes=True,
                host="127.0.0.1",
                dashboard_address=None, worker_dashboard_address=None,
                silence_logs=logging.ERROR,
                scheduler_kwargs={"allowed_failures": 0, "dashboard_address": "127.0.0.1:0"},
                death_timeout=self.startup_timeout, memory_limit=memory_limit)
            self.client = Client(self.cluster, set_as_default=False, timeout=self.startup_timeout)
            self.client.register_plugin(IndustrialWorkerBootstrap(catalog_json))
            self.cluster.scale(self.max_workers)
            while True:
                if self.cancellation.cancelled:
                    raise InterruptedError("Dask startup cancelled")
                remaining = self.startup_timeout - (monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Dask worker startup deadline reached")
                try:
                    self.client.wait_for_workers(self.max_workers, timeout=min(.2, remaining))
                    break
                except TimeoutError:
                    continue
            self.available_workers(())
            return self
        except BaseException:
            self.close()
            raise

    def available_workers(self, busy):
        from .dask_bootstrap import worker_resources

        info = self.client.scheduler_info()["workers"]
        self.worker_addresses = set(info)
        unchecked = sorted(set(info) - self._verified)
        if unchecked:
            resources = self.client.run(worker_resources, workers=unchecked, on_error="return")
            for address, resource in resources.items():
                if isinstance(resource, Exception):
                    # A newly discovered worker may disappear before its probe.
                    continue
                if (resource["digest"] != self.digest or resource["threads"] != 1
                        or resource["memory_limit"] <= 0 or not info[address].get("nanny")):
                    raise ValueError("Worker resources or manifest are not suitable for Industrial evaluation")
                self.worker_pids.add(resource["pid"])
                self._verified.add(address)
        return sorted((set(info) & self._verified) - set(busy))

    def interrupt_worker(self, address, future):
        self.client.cancel(future, force=True)
        reply = self.client.restart_workers(
            [address], timeout=self.config.restart_timeout_seconds, raise_for_error=False)
        self._verified.discard(address)
        if reply.get(address) != "OK":
            raise RuntimeError("Cannot confirm interruption of an owned Dask worker")

    def close(self):
        # Attempt both even if one cleanup fails. Never close an ambient client.
        errors = []
        for name in ("client", "cluster"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    resource.close(timeout=self.config.restart_timeout_seconds)
                except Exception as error:
                    errors.append(error)
                else:
                    setattr(self, name, None)
        if errors:
            raise DaskCleanupError("Could not confirm Dask resource cleanup") from errors[0]

    def __exit__(self, *args):
        self.close()


def _receive(flight):
    try:
        status = flight.future.status
        if flight.result_path.exists():
            result = cloudpickle.loads(flight.result_path.read_bytes())
            if not isinstance(result, ExecutionResult) or result.candidate_id != flight.task.candidate_id:
                raise ValueError("Dask worker returned a mismatched execution result")
            return replace(result, duration_seconds=monotonic() - flight.started)
        if status == "finished":
            return _failed(flight.task, ExecutionStatus.WORKER_FAILED, flight.started,
                           "MissingDurableResult", "Dask receipt has no persisted execution result")
        if status in ("error", "cancelled"):
            # Never gather a potentially lost computed value. Even exception
            # retrieval must yield periodically to cancellation and deadlines.
            error = flight.future.exception(timeout=.1)
            return _failed(flight.task, ExecutionStatus.WORKER_FAILED, flight.started,
                           type(error).__name__, str(error))
        return None
    except TimeoutError:
        return None
    except Exception as error:
        return _failed(flight.task, ExecutionStatus.WORKER_FAILED, flight.started, type(error).__name__, str(error))


def _run(session, plan, payloads, evaluate, token, deadline, config, on_attempt, directory):
    from .dask_bootstrap import evaluate_on_worker

    pending = deque((task, payload, 0) for task, payload in zip(plan.tasks, payloads))
    active, results, first_started = {}, {}, {}
    run_id = uuid4().hex
    ordinal = 0
    no_workers_since = None
    try:
        while pending or active:
            reason = _stop_reason(token, deadline)
            if reason:
                for task, _, _ in pending:
                    results[task.candidate_id] = _failed(task, reason[0], monotonic(), *reason[1:])
                for uid, flight in active.items():
                    results[uid] = _failed(flight.task, reason[0], flight.started, *reason[1:])
                    if on_attempt is not None:
                        on_attempt(DaskAttempt(uid, flight.attempt + 1, flight.worker, reason[0]))
                break
            session.available_workers(flight.worker for flight in active.values())
            for uid, flight in tuple(active.items()):
                expired = (plan.candidate_timeout_seconds is not None
                           and monotonic() - flight.started >= plan.candidate_timeout_seconds)
                if expired:
                    result = _failed(flight.task, ExecutionStatus.TIMED_OUT, flight.started,
                                     "EvaluationTimeout", "Dask candidate deadline reached")
                    results[uid] = result
                    session.interrupt_worker(flight.worker, flight.future)
                elif flight.result_path.exists() or flight.future.done():
                    result = _receive(flight)
                    if result is None:
                        if flight.worker in session.worker_addresses:
                            continue
                        session.client.cancel(flight.future, force=True)
                        result = _failed(flight.task, ExecutionStatus.WORKER_FAILED, flight.started,
                                         "WorkerLost", "Computed Dask receipt was lost before delivery")
                    reason = _stop_reason(token, deadline)
                    if reason:
                        result = _failed(flight.task, reason[0], flight.started, *reason[1:])
                    elif (plan.candidate_timeout_seconds is not None
                          and monotonic() - flight.started >= plan.candidate_timeout_seconds):
                        result = _failed(flight.task, ExecutionStatus.TIMED_OUT, flight.started,
                                         "EvaluationTimeout", "Result arrived after the Dask candidate deadline")
                elif flight.worker not in session.worker_addresses:
                    # A pinned task lost before execution may remain pending in
                    # Dask rather than failing as KilledWorker. Bound it here too.
                    session.client.cancel(flight.future, force=True)
                    result = _failed(flight.task, ExecutionStatus.WORKER_FAILED, flight.started,
                                     "WorkerLost", "Assigned Dask worker disappeared")
                else:
                    continue
                if on_attempt is not None:
                    on_attempt(DaskAttempt(uid, flight.attempt + 1, flight.worker, result.status))
                del active[uid]
                if retry_worker_failure(result, flight.attempt, config.retries):
                    pending.appendleft((flight.task, flight.payload, flight.attempt + 1))
                else:
                    results[uid] = result
                flight.future.release()
            workers = session.available_workers(flight.worker for flight in active.values())
            if pending and not workers and not active:
                no_workers_since = monotonic() if no_workers_since is None else no_workers_since
                if monotonic() - no_workers_since >= config.startup_timeout_seconds:
                    raise TimeoutError("No verified Dask worker became available")
            else:
                no_workers_since = None
            while pending and workers and not _stop_reason(token, deadline):
                task, payload, attempt = pending.popleft()
                started = first_started.setdefault(task.candidate_id, monotonic())
                if (plan.candidate_timeout_seconds is not None
                        and monotonic() - started >= plan.candidate_timeout_seconds):
                    results[task.candidate_id] = _failed(
                        task, ExecutionStatus.TIMED_OUT, started,
                        "EvaluationTimeout", "Dask retry budget exhausted")
                    continue
                worker = workers.pop(0)
                result_path = directory / f"{ordinal}.result"
                try:
                    future = session.client.submit(
                        evaluate_on_worker, task, payload, evaluate, session.digest, str(result_path),
                        key=f"industrial-evaluation-{run_id}-{ordinal}", pure=False, retries=0,
                        workers=[worker], allow_other_workers=False)
                except Exception as error:
                    results[task.candidate_id] = _failed(
                        task, ExecutionStatus.FAILED, started, type(error).__name__, str(error))
                    if on_attempt is not None:
                        on_attempt(DaskAttempt(task.candidate_id, attempt + 1, worker, ExecutionStatus.FAILED))
                    workers.append(worker)
                    ordinal += 1
                    continue
                ordinal += 1
                active[task.candidate_id] = _Flight(task, payload, attempt, future, worker, started, result_path)
            if pending or active:
                sleep(config.poll_interval_seconds)
    except Exception as error:
        for task in plan.tasks:
            if task.candidate_id not in results:
                results[task.candidate_id] = _failed(
                    task, ExecutionStatus.WORKER_FAILED, first_started.get(task.candidate_id, monotonic()),
                    type(error).__name__, str(error))
    finally:
        for flight in active.values():
            session.client.cancel(flight.future, force=True)
            flight.future.release()
    return tuple(results[task.candidate_id] for task in plan.tasks)


def execute_dask_plan(plan, payloads, evaluate, *, config=None, cancellation=None, on_attempt=None):
    """Run an explicit private cluster; return one terminal result per task.

    Infrastructure retries are bounded and opt-in. Model exceptions are never
    retried. Startup counts against the total budget; candidate budgets start at
    first submission and never reset between attempts. All payloads are trusted
    local Python objects, not untrusted network requests.
    """
    if not isinstance(plan, DaskEvaluationPlan):
        raise TypeError("Dask execution requires DaskEvaluationPlan")
    config = DaskExecutionConfig() if config is None else config
    if not isinstance(config, DaskExecutionConfig):
        raise TypeError("config must be DaskExecutionConfig")
    payloads = tuple(payloads)
    if len(payloads) != len(plan.tasks):
        raise ValueError("Each planned task requires exactly one payload")
    if not plan.tasks:
        return ()
    token = CancellationToken() if cancellation is None else cancellation
    started = monotonic()
    deadline = None if plan.total_timeout_seconds is None else started + plan.total_timeout_seconds
    reason = _stop_reason(token, deadline)
    if reason:
        return tuple(_failed(task, reason[0], started, *reason[1:]) for task in plan.tasks)
    startup = config.startup_timeout_seconds
    if deadline is not None:
        startup = min(startup, max(0.001, deadline - monotonic()))
    try:
        with _result_directory() as directory:
            with DaskEvaluationSession(
                    plan.max_workers, config, startup_timeout=startup, cancellation=token) as session:
                return _run(session, plan, payloads, evaluate, token, deadline, config, on_attempt, directory)
    except DaskCleanupError:
        raise
    except Exception as error:
        # Startup failure has no successful model result. Cleanup failure must not
        # turn an unconfirmed teardown into a successful execution either.
        reason = _stop_reason(token, deadline)
        if reason:
            return tuple(_failed(task, reason[0], started, *reason[1:]) for task in plan.tasks)
        return tuple(_failed(task, ExecutionStatus.WORKER_FAILED, started,
                             type(error).__name__, str(error)) for task in plan.tasks)
