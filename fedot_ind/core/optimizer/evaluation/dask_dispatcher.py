"""Dask interpreter reusing the same graph preparation and result reducer."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from time import monotonic

from golem.utilities.utilities import determine_n_jobs

from .configuration import EvaluationBackend
from .dask_config import DaskEvaluationPlan, DaskLifecycle
from .dask_executor import DaskCleanupError, DaskEvaluationSession, execute_dask_plan, _failed
from .contracts import ExecutionStatus
from .dispatcher import LocalEvaluationDispatcher, _GraphEvaluator
from dataclasses import dataclass


@dataclass
class _SearchResources:
    stack: ExitStack
    session: DaskEvaluationSession | None = None


_SESSION = ContextVar('industrial_explicit_dask_session', default=None)
_OBSERVER = ContextVar('industrial_explicit_dask_observer', default=None)


@contextmanager
def dask_session_scope(session=None, *, observer=None):
    """Opt in to a caller-owned, already entered session and/or diagnostics.

    This scope never creates or closes a client. Without it each batch retains
    its original privately owned lifecycle. The caller must close its session.
    An arbitrary ambient distributed.Client is never discovered or adopted.
    """
    if session is not None and (not isinstance(session, DaskEvaluationSession) or session.client is None):
        raise ValueError('Enter the caller-owned DaskEvaluationSession first')
    if observer is not None and not callable(observer):
        raise TypeError('Observer must be a context-manager factory')
    session_token, observer_token = _SESSION.set(session), _OBSERVER.set(observer)
    try:
        yield session
    finally:
        _OBSERVER.reset(observer_token)
        _SESSION.reset(session_token)


class DaskEvaluationDispatcher(LocalEvaluationDispatcher):
    supported_backends = (EvaluationBackend.DASK,)

    @contextmanager
    def search_scope(self):
        """Own a lazy session for this search; an explicit caller session wins."""
        previous = getattr(self, '_search_resources', None)
        self.attempt_history = []
        with ExitStack() as stack:
            self._search_resources = _SearchResources(stack)
            try:
                yield
            finally:
                self._search_resources = previous

    def _search_session(self, plan):
        session = _SESSION.get()
        resources = getattr(self, '_search_resources', None)
        if session is not None or self.config.dask.lifecycle is DaskLifecycle.BATCH:
            return session
        if resources is None:
            raise RuntimeError('search lifecycle requires IndustrialEvoOptimizer.optimise or dispatcher.search_scope')
        if resources.session is None and plan.tasks:
            startup = self.config.dask.startup_timeout_seconds
            if plan.total_timeout_seconds is not None:
                startup = min(startup, plan.total_timeout_seconds)
            resources.session = resources.stack.enter_context(DaskEvaluationSession(
                plan.max_workers, self.config.dask, startup_timeout=startup, cancellation=self.cancellation))
            observer = _OBSERVER.get()
            if observer is not None:
                resources.stack.enter_context(observer(resources.session))
        return resources.session

    def _make_plan(self, tasks, total):
        return DaskEvaluationPlan(tasks, determine_n_jobs(self._n_jobs, self.logger),
                                  self.config.candidate_timeout_seconds, total)

    def _execute_plan(self, plan, jobs):
        attempts = []
        started = monotonic()
        try:
            try:
                session = self._search_session(plan)
            except DaskCleanupError:
                raise
            except Exception as error:
                expired = plan.total_timeout_seconds is not None and monotonic() - started >= plan.total_timeout_seconds
                return tuple(_failed(task, ExecutionStatus.TIMED_OUT if expired else ExecutionStatus.WORKER_FAILED,
                                     started, type(error).__name__, str(error)) for task in plan.tasks)
            if plan.total_timeout_seconds is not None:
                remaining = plan.total_timeout_seconds - (monotonic() - started)
                if remaining <= 0:
                    return tuple(_failed(task, ExecutionStatus.TIMED_OUT, started,
                                         'EvaluationTimeout', 'Session startup consumed the evaluation budget')
                                 for task in plan.tasks)
                plan = replace(plan, total_timeout_seconds=remaining)
            return execute_dask_plan(
                plan, jobs,
                _GraphEvaluator(self._adapter, self._objective_eval, self._cleanup, self._post_eval_callback),
                config=self.config.dask, cancellation=self.cancellation, on_attempt=attempts.append,
                session=session, on_session=(None if session is not None and _SESSION.get() is None
                                             else _OBSERVER.get()))
        finally:
            self.last_dask_attempts = tuple(attempts)
            if not hasattr(self, 'attempt_history'):
                self.attempt_history = []
            self.attempt_history.extend(attempts)
