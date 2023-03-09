import pathlib
from typing import Optional, Tuple, Sequence

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher, OptionalEvalResult, \
    SequentialDispatcher, BasePipelineEvaluationDispatcher
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from fedot.core.optimisers.populational_optimizer import _unfit_pipeline
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.utilities.grouped_condition import GroupedCondition

from core.architecture.utils.utils import PROJECT_PATH


class IndustrialDispatcher(MultiprocessingDispatcher):
    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit: bool = True,
                        cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> OptionalEvalResult:
        OperationTypesRepository.__repository_dict__.update({'data_operation':
                                                                 {'file': pathlib.Path(PROJECT_PATH, 'core', 'repository',
                                                                               'data',
                                                                               'data_operation_repository.json'),
                                                                  'initialized_repo': None,
                                                                  'default_tags': []}})
        OperationTypesRepository.init_default_repositories()
        eval_res = super().evaluate_single(graph, uid_of_individual, with_time_limit, cache_key, logs_initializer)
        return eval_res


def init_pop(self,
             objective: Objective,
             initial_graphs: Sequence[Graph],
             requirements: PipelineComposerRequirements,
             graph_generation_params: GraphGenerationParams,
             graph_optimizer_params: Optional['GraphOptimizerParameters'] = None,
             ):
    super(GraphOptimizer).__new__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
    self.population = None
    self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
    self.timer = OptimisationTimer(timeout=self.requirements.timeout)

    dispatcher_type = IndustrialDispatcher if self.requirements.parallelization_mode == 'populational' else \
        SequentialDispatcher

    self.eval_dispatcher = dispatcher_type(adapter=graph_generation_params.adapter,
                                           n_jobs=requirements.n_jobs,
                                           graph_cleanup_fn=_unfit_pipeline,
                                           delegate_evaluator=graph_generation_params.remote_evaluator)

    # early_stopping_iterations and early_stopping_timeout may be None, so use some obvious max number
    max_stagnation_length = requirements.early_stopping_iterations or requirements.num_of_generations
    max_stagnation_time = requirements.early_stopping_timeout or self.timer.timeout
    self.stop_optimization = \
        GroupedCondition(results_as_message=True).add_condition(
            lambda: self.timer.is_time_limit_reached(self.current_generation_num),
            'Optimisation stopped: Time limit is reached'
        ).add_condition(
            lambda: self.requirements.num_of_generations is not None and
                    self.current_generation_num >= self.requirements.num_of_generations + 1,
            'Optimisation stopped: Max number of generations reached'
        ).add_condition(
            lambda: self.generations.stagnation_iter_count >= max_stagnation_length,
            'Optimisation finished: Early stopping iterations criteria was satisfied'
        ).add_condition(
            lambda: self.generations.stagnation_time_duration >= max_stagnation_time,
            'Optimisation finished: Early stopping timeout criteria was satisfied'
        )