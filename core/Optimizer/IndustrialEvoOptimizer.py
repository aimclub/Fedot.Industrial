from typing import Sequence

from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.objective import Objective
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.populational_optimizer import _unfit_pipeline

from core.repository.IndustrialDispatcher import IndustrialDispatcher


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: PipelineComposerRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPGraphOptimizerParameters):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.eval_dispatcher = IndustrialDispatcher(adapter=graph_generation_params.adapter,
                                                    n_jobs=requirements.n_jobs,
                                                    graph_cleanup_fn=_unfit_pipeline,
                                                    delegate_evaluator=graph_generation_params.remote_evaluator)
