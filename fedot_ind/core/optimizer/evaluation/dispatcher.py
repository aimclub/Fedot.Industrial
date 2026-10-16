"""GOLEM boundary for explicit local execution, without Dask or hidden retries."""

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime
import gc
from math import isfinite
from time import monotonic
from threading import Lock

from golem.core.optimisers.fitness import Fitness, null_fitness
from golem.core.optimisers.genetic.evaluation import BaseGraphEvaluationDispatcher
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.utilities.utilities import determine_n_jobs

from fedot_ind.core.optimizer.domain import (
    EvaluationFailed, EvaluationFailure, EvaluationFailureCode, EvaluationSucceeded, evaluation_outcome_to_metadata,
)
from fedot_ind.core.optimizer.observability import EvaluationStatus, graph_identity
from fedot_ind.core.optimizer.randomness import CandidateRandomness
from fedot_ind.integration.fedot.evaluation import industrial_failure_from_fedot
from fedot_ind.integration.fedot.extensions import industrial_extension_scope
from .configuration import EvaluationBackend, EvaluationExecutionConfig
from .contracts import EvaluationPlan, EvaluationTask, ExecutionStatus, LocalBackend
from .diagnostics import record_evaluation_results
from .executors import CancellationToken, execute_plan


@dataclass(frozen=True)
class _GraphJob:
    graph: object
    candidate_id: str


def _finite_fitness(fitness: Fitness) -> bool:
    """GOLEM's validity flag alone admits NaN and infinity."""
    try:
        return fitness.valid and bool(fitness.values) and all(isfinite(value) for value in fitness.values)
    except (TypeError, ValueError):
        return False


@dataclass
class _GraphEvaluator:
    adapter: object
    objective: object
    cleanup: object
    callback: object

    def _evaluate_domain(self, graph):
        try:
            fitness = self.objective(graph)
            if self.callback is not None:
                self.callback(graph)
            return fitness, graph
        finally:
            try:
                if self.cleanup is not None:
                    self.cleanup(graph)
            finally:
                gc.collect()

    def __call__(self, job):
        started = monotonic()
        # Every spawn process bootstraps its own extension scope before restoration.
        with industrial_extension_scope():
            fitness, graph = self.adapter.adapt_func(self._evaluate_domain)(job.graph)
            duration, timestamp = monotonic() - started, datetime.now().isoformat()
            if _finite_fitness(fitness):
                outcome = EvaluationSucceeded(duration, timestamp)
            else:
                owner = getattr(self.objective, '__self__', None)
                failure = industrial_failure_from_fedot(getattr(owner, 'last_outcome', None))
                outcome = EvaluationFailed(duration, timestamp, failure or EvaluationFailure(
                    EvaluationFailureCode.INVALID_FITNESS, 'InvalidFitness',
                    'Objective returned invalid or nonfinite fitness'))
                fitness = null_fitness()
        return GraphEvalResult(job.candidate_id, fitness, graph, evaluation_outcome_to_metadata(outcome))


class LocalEvaluationDispatcher(BaseGraphEvaluationDispatcher):
    """Own one batch at a time. No callback is executed in a shared thread."""

    allows_untimed_retry = False

    def __init__(self, adapter, n_jobs=1, graph_cleanup_fn=None, delegate_evaluator=None,
                 diagnostics_recorder=None, *, config: EvaluationExecutionConfig, random_seed=None):
        if not isinstance(config, EvaluationExecutionConfig) or config.backend is EvaluationBackend.LEGACY:
            raise ValueError('Local dispatcher requires an explicit local execution config')
        if delegate_evaluator is not None and delegate_evaluator.is_enabled:
            raise ValueError('Remote delegation requires a separate execution strategy')
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self.config = config
        self.random_seed = random_seed
        self.diagnostics_recorder = diagnostics_recorder
        self.cancellation = CancellationToken()
        self._evaluation_lock = Lock()
        self.last_plan = None
        self.last_results = ()

    def cancel(self):
        self.cancellation.cancel()

    def dispatch(self, objective, timer=None):
        if self._evaluation_lock.locked():
            raise RuntimeError('Cannot replace objective while a batch is active')
        super().dispatch(objective, timer)
        self.cancellation = CancellationToken()
        return self.evaluate_with_cache

    def evaluate_with_cache(self, population):
        with self._owned_batch():
            try:
                self._remote_compute_cache(population)
                return self._evaluate_population(population)
            finally:
                self._reset_eval_cache()

    def evaluate_population(self, individuals):
        with self._owned_batch():
            return self._evaluate_population(individuals)

    @contextmanager
    def _owned_batch(self):
        if not self._evaluation_lock.acquire(blocking=False):
            raise RuntimeError('Local dispatcher already owns an active batch')
        try:
            yield
        finally:
            self._evaluation_lock.release()

    def _evaluate_population(self, individuals):
        # A repeated reference to the same individual is evaluated/recorded once.
        unique = {}
        for individual in individuals:
            previous = unique.setdefault(str(individual.uid), individual)
            if previous is not individual:
                raise ValueError('Different candidates cannot share an individual uid')
            if individual.fitness.valid and not _finite_fitness(individual.fitness):
                raise ValueError('Existing candidate has nonfinite fitness; clear it before evaluation')
        to_evaluate, reused = self.split_individuals_to_evaluate(list(unique.values()))
        backend = LocalBackend(self.config.backend.value)
        total = self.config.total_timeout_seconds
        if self.timer.timeout is not None:
            remaining = self.timer.timeout.total_seconds() - self.timer.seconds_from_start
            if remaining <= 0:
                # An expired budget produces explicit timeout records, not an unbounded retry.
                total = 1e-12
            else:
                total = remaining if total is None else min(total, remaining)
        tasks = tuple(
            EvaluationTask(str(ind.uid), None if self.random_seed is None else CandidateRandomness(
                self.random_seed, 'evaluation:' + graph_identity(ind.graph), 0).seed)
            for ind in to_evaluate)
        plan = EvaluationPlan(tasks, backend,
                              1 if backend is LocalBackend.SEQUENTIAL else determine_n_jobs(self._n_jobs, self.logger),
                              self.config.candidate_timeout_seconds, total)
        jobs = tuple(_GraphJob(self.evaluation_cache.get(ind.uid, ind.graph), str(ind.uid)) for ind in to_evaluate)
        self.last_plan = plan
        self.last_results = execute_plan(
            plan, jobs, _GraphEvaluator(self._adapter, self._objective_eval, self._cleanup, self._post_eval_callback),
            cancellation=self.cancellation)
        graph_results = []
        for result, individual in zip(self.last_results, to_evaluate):
            if result.status is ExecutionStatus.SUCCEEDED:
                graph_result = result.value
            else:
                code = {ExecutionStatus.CANCELLED: EvaluationFailureCode.CANCELLED,
                        ExecutionStatus.TIMED_OUT: EvaluationFailureCode.TIMED_OUT,
                        ExecutionStatus.WORKER_FAILED: EvaluationFailureCode.WORKER_FAILED,
                        ExecutionStatus.FAILED: EvaluationFailureCode.OBJECTIVE_EXCEPTION}[result.status]
                outcome = EvaluationFailed(result.duration_seconds, datetime.now().isoformat(), EvaluationFailure(
                    code, result.error.error_type, result.error.message))
                graph_result = GraphEvalResult(
                    individual.uid, null_fitness(), individual.graph, evaluation_outcome_to_metadata(outcome))
            graph_results.append(graph_result)
        record_evaluation_results(self.diagnostics_recorder, graph_results)
        if self.diagnostics_recorder is not None:
            for individual in reused:
                self.diagnostics_recorder.record_evaluation(
                    individual_id=str(individual.uid), graph_id=graph_identity(individual.graph),
                    status=EvaluationStatus.REUSED)
        # GOLEM's base method also applies invalid fitness. Keep failed individuals
        # out of the breeding population, while retaining their typed diagnostics.
        successful = [r for r in graph_results if r.fitness.valid]
        evaluated = self.apply_evaluation_results(to_evaluate, successful)
        return evaluated + reused
