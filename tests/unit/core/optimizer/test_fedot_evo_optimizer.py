from types import SimpleNamespace

from golem.core.optimisers.adaptive.operator_agent import RandomAgent

from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    MutationAgentType,
    MutationStrategy,
)
from fedot_ind.core.optimizer.FedotEvoOptimizer import FedotEvoOptimizer
from fedot_ind.core.repository.constanst_repository import FEDOT_MUTATION_STRATEGY


def test_fedot_optimizer_builds_mutation_agent_from_typed_configuration():
    optimizer = FedotEvoOptimizer.__new__(FedotEvoOptimizer)
    optimizer.mutation_agent_dict = {MutationAgentType.RANDOM: RandomAgent}
    graph_optimizer_params = SimpleNamespace(mutation_types=[])
    config = EvolutionConfig(
        mutation_agent=MutationAgentType.RANDOM,
        mutation_strategy=MutationStrategy.GROWTH,
    )

    agent = optimizer._set_optimisation_strategy(graph_optimizer_params, config)

    assert isinstance(agent, RandomAgent)
    assert agent._probs == FEDOT_MUTATION_STRATEGY[MutationStrategy.GROWTH.value]
