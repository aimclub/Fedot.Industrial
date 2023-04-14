from typing import Sequence

from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph

from core.repository.IndustrialDispatcher import IndustrialDispatcher
from core.repository.initializer_industrial_models import add_preprocessing


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters):
        graph_optimizer_params.mutation_types.append(add_preprocessing)
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.eval_dispatcher = IndustrialDispatcher(adapter=graph_generation_params.adapter,
                                                    n_jobs=requirements.n_jobs,
                                                    graph_cleanup_fn=_try_unfit_graph,
                                                    delegate_evaluator=graph_generation_params.remote_evaluator)