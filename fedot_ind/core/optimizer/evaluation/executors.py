"""Owned local execution. Process deadlines include startup and result delivery."""

from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep

import cloudpickle

from fedot_ind.core.optimizer.random_scope import operator_random_scope as _random_scope
from .contracts import (
    EvaluationPlan, ExecutionError, ExecutionResult, ExecutionStatus, LocalBackend,
)


class CancellationToken:
    """Coordinator-owned cancellation; workers are stopped by their owner."""

    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


def _failure(task, status, started, error_type, message):
    return ExecutionResult(task.candidate_id, status, monotonic() - started,
                           error=ExecutionError(error_type, message))


def _invoke(task, payload, evaluate):
    started = monotonic()
    try:
        with _random_scope(task.seed):
            value = evaluate(payload)
        return ExecutionResult(task.candidate_id, ExecutionStatus.SUCCEEDED, monotonic() - started, value=value)
    except Exception as error:
        return _failure(task, ExecutionStatus.FAILED, started, type(error).__name__, str(error))


def _worker(input_path, result_path):
    task, payload, evaluate = cloudpickle.loads(Path(input_path).read_bytes())
    temporary = Path(result_path).with_suffix('.partial')
    temporary.write_bytes(cloudpickle.dumps(_invoke(task, payload, evaluate)))
    temporary.replace(result_path)


@dataclass
class _Running:
    task: object
    process: object
    result_path: Path
    started: float
    input_path: Path


def _remove_job_files(input_path, result_path):
    for path in (input_path, result_path, result_path.with_suffix('.partial')):
        path.unlink(missing_ok=True)


def _dispose(job):
    """Reap every owned child, including interrupted and broken workers."""
    if job.process.pid is None:
        job.process.close()
        _remove_job_files(job.input_path, job.result_path)
        return
    try:
        if job.process.is_alive():
            job.process.terminate()
        job.process.join(timeout=2)
        if job.process.is_alive():
            job.process.kill()
            job.process.join(timeout=2)
        if job.process.is_alive():
            raise RuntimeError(f"Cannot reap evaluation worker {job.process.pid}")
    finally:
        if not job.process.is_alive():
            job.process.close()
            _remove_job_files(job.input_path, job.result_path)


def _stop_reason(token, deadline):
    if token.cancelled:
        return ExecutionStatus.CANCELLED, "Cancelled", "Evaluation cancelled by coordinator"
    if deadline is not None and monotonic() >= deadline:
        return ExecutionStatus.TIMED_OUT, "EvaluationTimeout", "Total evaluation deadline reached"
    return None


def _receive(job):
    try:
        published = job.result_path.exists()
        exited = False if published else not job.process.is_alive()
        # Publication may occur between the first stat and observing exit.
        if published or (exited and job.result_path.exists()):
            result = cloudpickle.loads(job.result_path.read_bytes())
            if not isinstance(result, ExecutionResult) or result.candidate_id != job.task.candidate_id:
                raise ValueError("Worker returned a mismatched evaluation result")
            return result
        if exited:
            return _failure(job.task, ExecutionStatus.WORKER_FAILED, job.started,
                            "WorkerExited", f"Worker exit code {job.process.exitcode}")
    except Exception as error:
        return _failure(job.task, ExecutionStatus.WORKER_FAILED, job.started, type(error).__name__, str(error))
    return None


def _sequential(plan, payloads, evaluate, token, deadline):
    results = []
    for task, payload in zip(plan.tasks, payloads):
        started = monotonic()
        reason = _stop_reason(token, deadline)
        if reason:
            results.append(_failure(task, reason[0], started, *reason[1:]))
            continue
        result = _invoke(task, payload, evaluate)
        reason = _stop_reason(token, deadline)
        if reason:
            result = _failure(task, reason[0], started, *reason[1:])
        elif plan.candidate_timeout_seconds is not None and monotonic() - started >= plan.candidate_timeout_seconds:
            result = _failure(task, ExecutionStatus.TIMED_OUT, started, "EvaluationTimeout",
                              "Sequential callback exceeded its cooperative time limit")
        results.append(result)
    return tuple(results)


def _processes(plan, payloads, evaluate, token, deadline):
    # Publish only complete results. A half-written pipe message must never
    # block the coordinator and prevent timeout/cancellation handling.
    with TemporaryDirectory(prefix='industrial-evaluation-') as directory:
        return _run_processes(plan, payloads, evaluate, token, deadline, Path(directory))


def _run_processes(plan, payloads, evaluate, token, deadline, directory):
    context = multiprocessing.get_context("spawn")
    pending = deque(zip(plan.tasks, payloads))
    active = {}
    results = {}
    ordinal = 0
    try:
        while pending or active:
            reason = _stop_reason(token, deadline)
            if reason:
                while pending:
                    task, _ = pending.popleft()
                    results[task.candidate_id] = _failure(task, reason[0], monotonic(), *reason[1:])
                for uid, job in tuple(active.items()):
                    results[uid] = _failure(job.task, reason[0], job.started, *reason[1:])
                    _dispose(job)
                    del active[uid]
                break
            while pending and len(active) < plan.max_workers:
                if _stop_reason(token, deadline):
                    break
                task, payload = pending.popleft()
                started = monotonic()
                input_path = directory / f'{ordinal}.input'
                result_path = directory / f'{ordinal}.result'
                ordinal += 1
                process = None
                try:
                    input_path.write_bytes(cloudpickle.dumps((task, payload, evaluate)))
                    reason = _stop_reason(token, deadline)
                    if reason:
                        results[task.candidate_id] = _failure(task, reason[0], started, *reason[1:])
                        _remove_job_files(input_path, result_path)
                        continue
                    process = context.Process(target=_worker, args=(str(input_path), str(result_path)))
                    active[task.candidate_id] = _Running(task, process, result_path, started, input_path)
                    process.start()
                except Exception as error:
                    if process is not None:
                        _dispose(active.pop(task.candidate_id))
                    else:
                        _remove_job_files(input_path, result_path)
                    results[task.candidate_id] = _failure(
                        task, ExecutionStatus.WORKER_FAILED, started, type(error).__name__, str(error))
                    continue
            for uid, job in tuple(active.items()):
                result = None
                # Check the deadline before accepting a late worker message.
                if _stop_reason(token, deadline):
                    break
                if (plan.candidate_timeout_seconds is not None
                        and monotonic() - job.started >= plan.candidate_timeout_seconds):
                    result = _failure(job.task, ExecutionStatus.TIMED_OUT, job.started,
                                      "EvaluationTimeout", "Candidate deadline reached")
                else:
                    result = _receive(job)
                # Deserialization and local I/O also consume the budget. Never
                # publish a late success or one delivered after cancellation.
                reason = _stop_reason(token, deadline)
                if reason:
                    result = _failure(job.task, reason[0], job.started, *reason[1:])
                elif (plan.candidate_timeout_seconds is not None
                      and monotonic() - job.started >= plan.candidate_timeout_seconds):
                    result = _failure(job.task, ExecutionStatus.TIMED_OUT, job.started,
                                      "EvaluationTimeout", "Candidate deadline reached")
                if result is not None:
                    results[uid] = result
                    _dispose(job)
                    del active[uid]
            if active:
                sleep(0.01)
    finally:
        with ExitStack() as resources:
            for job in active.values():
                resources.callback(_dispose, job)
    return tuple(results[task.candidate_id] for task in plan.tasks)


def execute_plan(plan: EvaluationPlan, payloads, evaluate, *, cancellation=None):
    """Execute once in plan order, not completion order; never retry failures.

    Sequential deadlines/cancellation are cooperative at callback boundaries.
    Use PROCESS for interrupting a running fit. No thread backend is exposed.
    Only trusted local Python payloads may cross this cloudpickle boundary.
    """
    payloads = tuple(payloads)
    if len(payloads) != len(plan.tasks):
        raise ValueError("Each planned task requires exactly one payload")
    token = cancellation if cancellation is not None else CancellationToken()
    deadline = None if plan.total_timeout_seconds is None else monotonic() + plan.total_timeout_seconds
    runner = _sequential if plan.backend is LocalBackend.SEQUENTIAL else _processes
    return runner(plan, payloads, evaluate, token, deadline)
