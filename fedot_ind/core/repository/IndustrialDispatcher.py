from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher

from fedot_ind.core.architecture.abstraction.decorators import DaskServer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

import logging
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple
from golem.core.log import Log
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.timer import Timer
from golem.utilities.memory import MemoryAnalytics
from golem.utilities.utilities import determine_n_jobs
from joblib import wrap_non_picklable_objects, Parallel, delayed, parallel_backend


class IndustrialDispatcher(MultiprocessingDispatcher):

    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        return self.evaluate_with_cache

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(
            individuals)

        # Evaluate individuals without valid fitness in parallel.
        n_jobs = determine_n_jobs(self._n_jobs, self.logger)
        parallel = Parallel(n_jobs=n_jobs, verbose=0,
                            pre_dispatch='2 * n_jobs')

        with parallel_backend(backend='dask',
                              n_jobs=n_jobs
                              # ,scatter=[individuals_to_evaluate]
                              ):
            evaluation_results = []
            for ind in individuals_to_evaluate:
                y = self.industrial_evaluate_single(self, graph=ind.graph, uid_of_individual=ind.uid,
                                                    logs_initializer=Log().get_parameters())
                evaluation_results.append(y)
            individuals_evaluated = self.apply_evaluation_results(
                individuals_to_evaluate, evaluation_results)
            # If there were no successful evals then try once again getting at least one,
            # even if time limit was reached
            successful_evals = individuals_evaluated + individuals_to_skip
            self.population_evaluation_info(evaluated_pop_size=len(successful_evals),
                                            pop_size=len(individuals))
            if not successful_evals:
                for single_ind in individuals:
                    try:
                        evaluation_result = self.industrial_evaluate_single(self, graph=ind.graph,
                                                                            uid_of_individual=ind.uid,
                                                                            with_time_limit=False)
                        successful_evals = self.apply_evaluation_results(
                            [single_ind], [evaluation_result])
                        if successful_evals:
                            break
                    except Exception:
                        _ = 1
            MemoryAnalytics.log(self.logger,
                                additional_info='parallel evaluation of population',
                                logging_level=logging.INFO)
        return successful_evals

    # @delayed
    @wrap_non_picklable_objects
    def industrial_evaluate_single(self,
                                   graph: OptGraph,
                                   uid_of_individual: str,
                                   with_time_limit: bool = True,
                                   cache_key: Optional[str] = None,
                                   logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> GraphEvalResult:
        if self._n_jobs != 1:
            OperationTypesRepository = IndustrialModels().setup_repository()

        graph = self.evaluation_cache.get(cache_key, graph)

        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso
            }
        )
        return eval_res
