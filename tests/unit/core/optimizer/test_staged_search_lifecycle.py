"""Synthetic objective integration test, not a trained model or quality experiment."""
from datetime import timedelta
from unittest.mock import patch

import numpy as np


def test_staged_evolution_completes_with_a_synthetic_objective_and_zero_fit_calls():
    from fedot import create_data
    from fedot.core.operations.operation import Operation
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from benchmark.industrial.experiments.staged_mutations.config import load_protocol
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy, ExecutionPolicy
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationExecutionConfig, EvaluationBackend
    from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
    from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
    from fedot_ind.integration.fedot.evaluation import ConfirmedObjective

    requirements, params, graph = _environment(dict(name='synthetic', task='regression',
                                                    operations=['scaling', 'normalization', 'ridge'], initial_chain=['scaling', 'ridge']),
                                               dict(max_depth=5, max_arity=2))
    requirements.timeout = timedelta(seconds=30)
    requirements.num_of_generations = 2
    policy = load_staged_search_defaults()
    algorithm = GPAlgorithmParameters(pop_size=4, max_pop_size=4, crossover_prob=0., mutation_prob=1.,
                                      mutation_types=[MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop])
    preparation = ParameterSearchPreparation.from_defaults(params.adapter, load_protocol()['defaults'],
                                                           policy.parameters, 'regression')
    optimizer = IndustrialEvoOptimizer(Objective({'loss': lambda graph: 0.}), [graph], requirements, params, algorithm,
                                       EvolutionConfig(random_seed=42, mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=32),
                                                       mutation_schedule=policy, execution_policy=ExecutionPolicy(
                                                           retry_initial_population_without_timer=False),
                                                       evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL, total_timeout_seconds=30)),
                                       evaluation_graph_preparation=preparation)
    data = create_data(np.ones((6, 2)), target=np.arange(6), task='regression')
    synthetic = ConfirmedObjective.from_training_data(lambda graph: SingleObjFitness(float(len(graph.nodes))), (data,))
    with patch.object(Operation, 'fit', side_effect=AssertionError('No training in this contract test')) as fit:
        selected = optimizer.optimise(synthetic)
        assert fit.call_count == 0
    assert selected and optimizer.completed_evaluations.entries
    assert 1 <= len(optimizer.staged_search.families) <= policy.max_families
    assert optimizer.staged_search.events[0]['name'] == 'explore'
