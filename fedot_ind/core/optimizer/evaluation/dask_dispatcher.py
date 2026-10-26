"""Dask interpreter reusing the same graph preparation and result reducer."""

from golem.utilities.utilities import determine_n_jobs

from .configuration import EvaluationBackend
from .dask_config import DaskEvaluationPlan
from .dask_executor import execute_dask_plan
from .dispatcher import LocalEvaluationDispatcher, _GraphEvaluator


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
                config=self.config.dask, cancellation=self.cancellation, on_attempt=attempts.append)
        finally:
            self.last_dask_attempts = tuple(attempts)
