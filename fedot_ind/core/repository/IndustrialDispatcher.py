import logging
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple

import dask
from golem.core.log import Log
from golem.core.optimisers.fitness import null_fitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.timer import Timer
from golem.utilities.memory import MemoryAnalytics
from golem.utilities.utilities import determine_n_jobs
from joblib import wrap_non_picklable_objects

from fedot_ind.core.optimizer.observability import (
    EvaluationStatus,
    EvolutionDiagnosticsRecorder,
    graph_identity,
)
from fedot_ind.integration.fedot.extensions import industrial_extension_scope


class IndustrialDispatcher(MultiprocessingDispatcher):

    def __init__(
            self,
            adapter,
            n_jobs: int = 1,
            graph_cleanup_fn=None,
            delegate_evaluator=None,
            diagnostics_recorder: Optional[EvolutionDiagnosticsRecorder] = None,
    ):
        super().__init__(
            adapter=adapter,
            n_jobs=n_jobs,
            graph_cleanup_fn=graph_cleanup_fn,
            delegate_evaluator=delegate_evaluator,
        )
        self.diagnostics_recorder = diagnostics_recorder

    def __getstate__(self):
        """Keep the coordinator-local recorder out of Dask worker payloads."""
        state = dict(self.__dict__)
        state['diagnostics_recorder'] = None
        return state

    def dispatch(self, objective: ObjectiveFunction,
                 timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        return self.evaluate_with_cache

    def _multithread_eval(self, individuals_to_evaluate):
        log = Log().get_parameters()
        evaluation_results = list(map(lambda ind:
                                      self.industrial_evaluate_single(self,
                                                                      graph=ind.graph,
                                                                      uid_of_individual=ind.uid,
                                                                      logs_initializer=log),
                                      individuals_to_evaluate))
        evaluation_results = dask.compute(*evaluation_results)
        return evaluation_results

    def _eval_at_least_one(self, individuals):
        successful_evals = []
        for single_ind in individuals:
            try:
                delayed_result = self.industrial_evaluate_single(
                    self,
                    graph=single_ind.graph,
                    uid_of_individual=single_ind.uid,
                    with_time_limit=False,
                )
                evaluation_result = dask.compute(delayed_result)[0]
                self._record_evaluation_results((evaluation_result,))
                successful_evals = self.apply_evaluation_results(
                    [single_ind], [evaluation_result])
                if successful_evals:
                    break
            except Exception:
                successful_evals = []
        return successful_evals

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)

        # Evaluate individuals without valid fitness in parallel.
        self.n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        evaluation_results = (
            self._multithread_eval(individuals_to_evaluate)
            if individuals_to_evaluate
            else []
        )
        self._record_evaluation_results(evaluation_results)
        self._record_reused_individuals(individuals_to_skip)
        individuals_evaluated = self.apply_evaluation_results(
            individuals_to_evaluate,
            evaluation_results,
        )

        successful_evals = individuals_evaluated + individuals_to_skip
        self.population_evaluation_info(evaluated_pop_size=len(successful_evals), pop_size=len(individuals))
        if not successful_evals:
            self._log_evaluation_failures(evaluation_results)
            successful_evals = self._eval_at_least_one(individuals)

        MemoryAnalytics.log(self.logger, additional_info='parallel evaluation of population',
                            logging_level=logging.INFO)
        return successful_evals

    def _record_evaluation_results(self, evaluation_results) -> None:
        recorder = getattr(self, 'diagnostics_recorder', None)
        if recorder is None:
            return
        for result in evaluation_results:
            if result is None:
                continue
            metadata = dict(result.metadata or {})
            error_message = metadata.get('evaluation_error')
            recorder.record_evaluation(
                individual_id=str(result.uid_of_individual),
                graph_id=graph_identity(result.graph),
                status=(
                    EvaluationStatus.FAILED if error_message else EvaluationStatus.SUCCEEDED),
                duration_seconds=metadata.get('computation_time_in_seconds'),
                error_type=metadata.get('evaluation_error_type'),
                error_message=error_message,
            )

    def _record_reused_individuals(self, individuals) -> None:
        recorder = getattr(self, 'diagnostics_recorder', None)
        if recorder is None:
            return
        for individual in individuals:
            recorder.record_evaluation(
                individual_id=str(individual.uid),
                graph_id=graph_identity(individual.graph),
                status=EvaluationStatus.REUSED,
            )

    def _log_evaluation_failures(self, evaluation_results) -> None:
        failures = [
            result.metadata.get('evaluation_error')
            for result in evaluation_results
            if result is not None and result.metadata.get('evaluation_error')
        ]
        for message in tuple(dict.fromkeys(failures))[:3]:
            self.logger.warning('Industrial graph evaluation failed: %s', message)

    @dask.delayed
    def eval_ind(self, graph, uid_of_individual):
        start_time = timeit.default_timer()
        evaluation_error = None
        evaluation_error_type = None
        try:
            with industrial_extension_scope():
                adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
                fitness, graph = adapted_evaluate(graph)
        except Exception as ex:
            self.logger.info(f'Graph evaluation failed. Assigning null fitness. Exception - {ex}')
            fitness = null_fitness()
            evaluation_error = repr(ex)
            evaluation_error_type = type(ex).__name__
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()
        if evaluation_error is None and not fitness.valid:
            evaluation_error = 'Objective returned invalid fitness without raising an exception'
            evaluation_error_type = 'InvalidFitness'
        metadata = {
            'computation_time_in_seconds': end_time - start_time,
            'evaluation_time_iso': eval_time_iso}
        if evaluation_error is not None:
            metadata['evaluation_error'] = evaluation_error
            metadata['evaluation_error_type'] = evaluation_error_type
        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual,
            fitness=fitness,
            graph=graph,
            metadata=metadata)
        return eval_res

    @wrap_non_picklable_objects
    def industrial_evaluate_single(self,
                                   graph: OptGraph,
                                   uid_of_individual: str,
                                   with_time_limit: bool = True,
                                   cache_key: Optional[str] = None,
                                   logs_initializer: Optional[Tuple[int,
                                                                    pathlib.Path]] = None) -> GraphEvalResult:
        graph = self.evaluation_cache.get(cache_key, graph)
        #
        # if with_time_limit and self.timer.is_time_limit_reached():
        #     return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        return self.eval_ind(graph, uid_of_individual)
