from typing import Any, Mapping, Sequence

from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    MutationAgentType,
    normalize_evolution_config,
)
from fedot_ind.core.optimizer.mutation import without_resample_mutations
from fedot_ind.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY


class FedotEvoOptimizer(EvoGraphOptimizer):
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 optimisation_params: Mapping[str, Any] | EvolutionConfig | None = None):

        graph_optimizer_params = self._exclude_resample_from_mutations(graph_optimizer_params)
        self.mutation_agent_dict = {
            MutationAgentType.RANDOM: RandomAgent,
            MutationAgentType.BANDIT: MultiArmedBanditAgent,
            MutationAgentType.CONTEXTUAL_BANDIT: ContextualMultiArmedBanditAgent,
            MutationAgentType.NEURAL_BANDIT: NeuralContextualMultiArmedBanditAgent,
        }
        if optimisation_params is not None:
            evolution_config = normalize_evolution_config(optimisation_params)
            graph_optimizer_params.adaptive_mutation_type = self._set_optimisation_strategy(graph_optimizer_params,
                                                                                            evolution_config)
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params)
        self.requirements = requirements
        # self.eval_dispatcher = IndustrialDispatcher(
        #     adapter=graph_generation_params.adapter,
        #     n_jobs=requirements.n_jobs,
        #     graph_cleanup_fn=_try_unfit_graph,
        #     delegate_evaluator=graph_generation_params.remote_evaluator)

    def _set_optimisation_strategy(self, graph_optimizer_params, config: EvolutionConfig):
        mutation_probs = FEDOT_MUTATION_STRATEGY[config.mutation_strategy.value]
        mutation_agent = self.mutation_agent_dict[config.mutation_agent]
        if config.mutation_agent is MutationAgentType.RANDOM:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types,
                                            probs=mutation_probs)
        else:
            mutation_agent = mutation_agent(actions=graph_optimizer_params.mutation_types)
        return mutation_agent

    def _exclude_resample_from_mutations(self, graph_optimizer_params):
        graph_optimizer_params.mutation_types = without_resample_mutations(
            graph_optimizer_params.mutation_types
        )
        return graph_optimizer_params
