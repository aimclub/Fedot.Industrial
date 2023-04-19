from typing import Sequence

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import _try_unfit_graph

from fedot_ind.core.repository.IndustrialDispatcher import IndustrialDispatcher
from fedot_ind.core.repository.initializer_industrial_models import \
    has_no_data_flow_conflicts_in_industrial_pipeline


class IndustrialEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters):

        graph_optimizer_params.mutation_types.remove(MutationTypesEnum.single_drop)
        graph_generation_params.verifier._rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        #graph_generation_params.verifier._rules.remove(has_no_conflicts_with_data_flow)

        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.eval_dispatcher = IndustrialDispatcher(adapter=graph_generation_params.adapter,
                                                    n_jobs=requirements.n_jobs,
                                                    graph_cleanup_fn=_try_unfit_graph,
                                                    delegate_evaluator=graph_generation_params.remote_evaluator)