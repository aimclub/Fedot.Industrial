"""Dask interpreter reusing the same graph preparation and result reducer."""

from contextlib import contextmanager
from contextvars import ContextVar

from golem.utilities.utilities import determine_n_jobs

from .configuration import EvaluationBackend
from .dask_config import DaskEvaluationPlan
from .dask_executor import DaskEvaluationSession, execute_dask_plan
from .dispatcher import LocalEvaluationDispatcher, _GraphEvaluator


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

    def _make_plan(self, tasks, total):
        return DaskEvaluationPlan(tasks, determine_n_jobs(self._n_jobs, self.logger),
                                  self.config.candidate_timeout_seconds, total)

    def _execute_plan(self, plan, jobs):
        attempts = []
        try:
            return execute_dask_plan(
                plan, jobs,
                _GraphEvaluator(self._adapter, self._objective_eval, self._cleanup, self._post_eval_callback),
                config=self.config.dask, cancellation=self.cancellation, on_attempt=attempts.append,
                session=_SESSION.get(), on_session=_OBSERVER.get())
        finally:
            self.last_dask_attempts = tuple(attempts)
