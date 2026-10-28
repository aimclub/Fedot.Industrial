"""GOLEM boundary for explicit local execution, without Dask or hidden retries."""

from dataclasses import dataclass, replace
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
from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import feature_reuse_scope
from .configuration import EvaluationBackend, EvaluationExecutionConfig
from .contracts import EvaluationPlan, EvaluationTask, ExecutionError, ExecutionResult, ExecutionStatus, LocalBackend
from .diagnostics import record_evaluation_results
from .executors import CancellationToken, execute_plan
from .context import EvaluationBudget, EvaluationContext, evaluation_context_scope
from .preparation import GraphPreparation, PreparedEvaluationGraph
from .timing import BatchTiming


@dataclass(frozen=True)
class _GraphJob:
    graph: object
    candidate_id: str
    context: EvaluationContext | None = None


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
        with industrial_extension_scope(), feature_reuse_scope(), evaluation_context_scope(job.context):
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
        metadata = evaluation_outcome_to_metadata(outcome)
        if job.context is not None:
            metadata['evaluation_context'] = job.context.to_record()
        return GraphEvalResult(job.candidate_id, fitness, graph, metadata)


class LocalEvaluationDispatcher(BaseGraphEvaluationDispatcher):
    """Own one batch at a time. No callback is executed in a shared thread."""

    allows_untimed_retry = False
    supported_backends = (EvaluationBackend.SEQUENTIAL, EvaluationBackend.PROCESS)

    def __init__(self, adapter, n_jobs=1, graph_cleanup_fn=None, delegate_evaluator=None,
                 diagnostics_recorder=None, *, config: EvaluationExecutionConfig, random_seed=None,
                 graph_preparation: GraphPreparation | None = None, completed_evaluation_callback=None):
        if not isinstance(config, EvaluationExecutionConfig) or config.backend not in self.supported_backends:
            raise ValueError('Local dispatcher requires an explicit local execution config')
        if delegate_evaluator is not None and delegate_evaluator.is_enabled:
            raise ValueError('Remote delegation requires a separate execution strategy')
        if graph_preparation is not None and not callable(graph_preparation):
            raise ValueError('Graph preparation must be callable or None')
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self.config = config
        self.random_seed = random_seed
        self.graph_preparation = graph_preparation
        self.completed_evaluation_callback = completed_evaluation_callback
        self._evaluation_sequence = 0
        self.diagnostics_recorder = diagnostics_recorder
        self.cancellation = CancellationToken()
        self._evaluation_lock = Lock()
        self.last_plan = None
        self.last_results = ()
        self.last_budget = None
        self.batch_timings = []

    def cancel(self):
        self.cancellation.cancel()

    def dispatch(self, objective, timer=None):
        if self._evaluation_lock.locked():
            raise RuntimeError('Cannot replace objective while a batch is active')
        super().dispatch(objective, timer)
        self.cancellation = CancellationToken()
        self.batch_timings = []
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
        preparation_started = monotonic()
        # A repeated reference to the same individual is evaluated/recorded once.
        unique = {}
        for individual in individuals:
            previous = unique.setdefault(str(individual.uid), individual)
            if previous is not individual:
                raise ValueError('Different candidates cannot share an individual uid')
            if individual.fitness.valid and not _finite_fitness(individual.fitness):
                raise ValueError('Existing candidate has nonfinite fitness; clear it before evaluation')
        to_evaluate, reused = self.split_individuals_to_evaluate(list(unique.values()))
        tasks, jobs = [], []
        for individual in to_evaluate:
            graph = self.evaluation_cache.get(individual.uid, individual.graph)
            prepared = self.graph_preparation(graph) if self.graph_preparation is not None else None
            if self.graph_preparation is not None and not isinstance(prepared, PreparedEvaluationGraph):
                raise TypeError('Graph preparation must return PreparedEvaluationGraph')
            identity = graph_identity(graph) if prepared is None else prepared.configuration_id
            seed = (None if self.random_seed is None else
                    CandidateRandomness(self.random_seed, 'evaluation:' + identity, 0).seed)
            candidate_id = str(individual.uid)
            context = EvaluationContext(
                f'{candidate_id}:{self._evaluation_sequence}', candidate_id, identity, seed,
                None if prepared is None else prepared.configuration_json)
            self._evaluation_sequence += 1
            tasks.append(EvaluationTask(candidate_id, seed))
            jobs.append(_GraphJob(graph if prepared is None else prepared.graph, candidate_id, context))
        tasks, jobs = tuple(tasks), tuple(jobs)
        # Preparation consumes the same search clock as model evaluation.
        preparation_seconds = monotonic() - preparation_started
        total = (None if self.config.total_timeout_seconds is None else
                 self.config.total_timeout_seconds - preparation_seconds)
        expired = total is not None and total <= 0
        remaining = None
        if self.timer.timeout is not None:
            remaining = self.timer.timeout.total_seconds() - self.timer.seconds_from_start
            if remaining <= 0:
                expired = True
            else:
                total = remaining if total is None else min(total, remaining)
        budget = EvaluationBudget(preparation_seconds, remaining, None if expired else total, expired)
        self.last_budget = budget
        jobs = tuple(replace(job, context=replace(job.context, budget=budget)) for job in jobs)
        plan = self._make_plan(tasks, None if expired else total)
        self.last_plan = plan
        execution_started = monotonic()
        if expired:
            # A tiny positive timeout can still admit a callback on a coarse host clock.
            cancelled = self.cancellation.cancelled
            self.last_results = tuple(ExecutionResult(
                task.candidate_id, ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.TIMED_OUT, 0.,
                error=ExecutionError('Cancelled' if cancelled else 'EvaluationTimeout',
                                     'Evaluation cancelled by coordinator' if cancelled
                                     else 'Total evaluation deadline reached before execution'))
                for task in tasks)
        else:
            self.last_results = self._execute_plan(plan, jobs)
        execution_seconds = monotonic() - execution_started
        application_started = monotonic()
        graph_results = []
        for result, individual, job in zip(self.last_results, to_evaluate, jobs):
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
                graph_result.metadata['evaluation_context'] = job.context.to_record()
            graph_results.append(graph_result)
        record_evaluation_results(self.diagnostics_recorder, graph_results)
        if self.completed_evaluation_callback is not None:
            self.completed_evaluation_callback(graph_results)
        if self.diagnostics_recorder is not None:
            for individual in reused:
                self.diagnostics_recorder.record_evaluation(
                    individual_id=str(individual.uid), graph_id=graph_identity(individual.graph),
                    status=EvaluationStatus.REUSED)
        # GOLEM's base method also applies invalid fitness. Keep failed individuals
        # out of the breeding population, while retaining their typed diagnostics.
        successful = [r for r in graph_results if r.fitness.valid]
        evaluated = self.apply_evaluation_results(to_evaluate, successful)
        self.batch_timings.append(BatchTiming(
            preparation_seconds, execution_seconds, monotonic() - application_started, len(tasks)))
        return evaluated + reused

    def _make_plan(self, tasks, total):
        backend = LocalBackend(self.config.backend.value)
        return EvaluationPlan(
            tasks, backend,
            1 if backend is LocalBackend.SEQUENTIAL else determine_n_jobs(self._n_jobs, self.logger),
            self.config.candidate_timeout_seconds, total)

    def _execute_plan(self, plan, jobs):
        return execute_plan(
            plan, jobs,
            _GraphEvaluator(self._adapter, self._objective_eval, self._cleanup, self._post_eval_callback),
            cancellation=self.cancellation)
