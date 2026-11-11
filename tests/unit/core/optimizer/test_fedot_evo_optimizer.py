from types import SimpleNamespace

import pytest
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.adaptive.operator_agent import RandomAgent
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.optimization_parameters import GraphRequirements

from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig,
    MutationAgentType,
    MutationStrategy,
)
from fedot_ind.core.optimizer.FedotEvoOptimizer import FedotEvoOptimizer
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.optimizer.randomness import SeededMutation
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


@pytest.mark.parametrize("optimizer_class", [FedotEvoOptimizer, IndustrialEvoOptimizer])
@pytest.mark.parametrize("seed", [None, 42])
def test_real_constructor_wires_seeded_operators_without_changing_default(optimizer_class, seed):
    optimizer = optimizer_class(
        objective=Objective({"size": lambda graph: len(graph.nodes)}),
        initial_graphs=[Pipeline(PrimaryNode("ridge"))],
        requirements=GraphRequirements(n_jobs=1, show_progress=False, keep_history=False),
        graph_generation_params=GraphGenerationParams(adapter=PipelineAdapter()),
        graph_optimizer_params=GPAlgorithmParameters(pop_size=2),
        optimisation_params=EvolutionConfig(random_seed=seed),
    )
    assert isinstance(optimizer.mutation, SeededMutation) is (seed is not None)
    assert optimizer.reproducer.mutation is optimizer.mutation
    assert optimizer.reproducer.selection is optimizer.inheritance.selection is optimizer.selection
    assert optimizer.reproducer.crossover is optimizer.crossover
    assert optimizer.mutation in optimizer.operators
