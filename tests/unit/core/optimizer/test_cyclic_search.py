from dataclasses import replace
from datetime import timedelta
import json

from hypothesis import given, strategies as st
import numpy as np
import pytest

from fedot_ind.core.optimizer.search_policy.cyclic import parse_cyclic_search
from fedot_ind.core.optimizer.search_policy.planning import stage_for_budget
from benchmark.industrial.experiments.cyclic_mutations.study import policy_for


@given(st.lists(st.floats(min_value=0, max_value=900, allow_nan=False), min_size=1, max_size=30))
def test_cyclic_clock_has_ordered_phases_and_reserved_refinement(times):
    policy = policy_for('cyclic')
    phases = [policy.stages.index(stage_for_budget(policy, t, 600)) for t in sorted(times)]
    assert phases == sorted(phases)
    for index, stage in enumerate(policy.stages):
        assert stage.conservative == bool(index % 2)
        assert stage.weights.to_record()['parameter_change'] == float(stage.conservative)


def test_policy_roundtrip_and_shared_domains():
    policy = policy_for('cyclic')
    assert parse_cyclic_search(json.loads(json.dumps(policy.to_record()))) == policy
    assert policy.parameters == policy_for('fixed').parameters
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy, normalize_evolution_config
    config = EvolutionConfig(mutation_schedule=policy, mutation_planning=MutationPlanningPolicy(enabled=True))
    assert normalize_evolution_config(config.to_record()) == config
    with pytest.raises(ValueError):
        replace(policy, active_fraction=1.)


def optimizer_fixture():
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy
    from fedot_ind.core.optimizer.evaluation.configuration import EvaluationExecutionConfig, EvaluationBackend
    from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
    requirements, params, graph = _environment(dict(name='synthetic', task='regression',
                                                    operations=['scaling', 'normalization', 'ridge', 'rf'], initial_chain=['scaling', 'ridge']), dict(max_depth=5, max_arity=2))
    requirements.timeout = timedelta(seconds=30)
    requirements.num_of_generations = 2
    policy = policy_for('cyclic')
    algorithm = GPAlgorithmParameters(pop_size=4, max_pop_size=4, crossover_prob=0., mutation_prob=1.,
                                      mutation_types=[MutationTypesEnum.single_change, MutationTypesEnum.single_add, MutationTypesEnum.single_drop])
    defaults = {'ridge': {'alpha': 1.}, 'logit': {'C': 1.}, 'rf': {'max_depth': None}}
    from fedot_ind.core.tuning.generator_catalog import generator_catalog
    defaults.update({s.operation: s.defaults for s in generator_catalog()})
    defaults['hankelisation']['window_size'] = 24
    preparation = ParameterSearchPreparation.from_defaults(params.adapter, defaults, policy.parameters,
                                                           'regression', normalize_generators=True)
    owner = IndustrialEvoOptimizer(Objective({'loss': lambda g: 0.}), [graph], requirements, params, algorithm,
                                   EvolutionConfig(random_seed=42, mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=32),
                                                   mutation_schedule=policy, evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.SEQUENTIAL,
                                                                                                                            total_timeout_seconds=30)),
                                   evaluation_graph_preparation=preparation)
    return owner, graph


def test_conservative_mutations_preserve_topology_and_frontier_exhausts_without_fit():
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.individual import Individual
    from fedot_ind.core.optimizer.mutation_runtime import project_for_mutation
    from fedot_ind.core.optimizer.mutation_planning._graph import candidate_signature
    from fedot_ind.core.optimizer.search_policy.batches import install_bounded_reproduction
    owner, graph = optimizer_fixture()
    parent = Individual(graph, fitness=SingleObjFitness(1.))
    owner.staged_search.remember(parent)
    install_bounded_reproduction(owner, 4)
    owner.reproducer.frontier.submitted.add(owner.reproducer.frontier.identity(parent.graph))
    owner.staged_search.apply(owner.mutation, 6.)  # first refinement, [5, 10) in this 30s test
    assert owner.staged_search.stage.conservative
    signature = candidate_signature(project_for_mutation(graph, owner.graph_generation_params))
    produced = []
    for _ in range(3):
        result = owner.mutation(parent)
        if result:
            produced.append(result)
            assert candidate_signature(project_for_mutation(result.graph, owner.graph_generation_params)) == signature
    assert len(produced) == 2  # alpha=.1 and 10; the current alpha=1 is not novel
    with pytest.raises(ValueError, match='backwards'):
        owner.staged_search.apply(owner.mutation, 1.)


def test_bounded_reproduction_runs_public_optimizer_with_no_model_fit():
    from unittest.mock import patch
    from fedot import create_data
    from golem.core.optimisers.fitness import SingleObjFitness
    from fedot_ind.core.optimizer.search_policy.batches import install_bounded_reproduction
    from fedot_ind.integration.fedot.evaluation import ConfirmedObjective
    owner, _ = optimizer_fixture()
    install_bounded_reproduction(owner, 2)
    data = create_data(np.ones((6, 2)), target=np.arange(6), task='regression')
    objective = ConfirmedObjective.from_training_data(lambda graph: SingleObjFitness(float(len(graph.nodes))), (data,))
    with patch('fedot.core.operations.operation.Operation.fit', side_effect=AssertionError('Fit forbidden')):
        assert owner.optimise(objective)
    assert owner.reproducer.batches and all(row['candidates'] <= 2 for row in owner.reproducer.batches)


def test_cycles_carry_improved_minima_for_multiple_families():
    from copy import deepcopy
    from golem.core.optimisers.fitness import SingleObjFitness
    from golem.core.optimisers.opt_history_objects.individual import Individual
    owner, graph = optimizer_fixture()
    first = Individual(graph, fitness=SingleObjFitness(2.))
    other_graph = deepcopy(graph)
    next(n for n in other_graph.nodes if n.content['name'] == 'scaling').content['name'] = 'normalization'
    second = Individual(other_graph, fitness=SingleObjFitness(3.))
    controller = owner.staged_search
    controller.remember(first)
    controller.remember(second)
    controller.apply(owner.mutation, 6.)
    assert len(controller.locked_families) == 2
    improved_graph = deepcopy(graph)
    next(n for n in improved_graph.nodes if n.content['name'] == 'ridge').content['params'] = {'alpha': 10.}
    improved = Individual(improved_graph, fitness=SingleObjFitness(1.))
    controller.remember(improved)
    controller.remember(Individual(graph, fitness=SingleObjFitness(4.)))
    assert sorted(i.fitness.value for i in controller.refinement_parents(2)) == [1., 3.]
    controller.apply(owner.mutation, 11.)
    controller.apply(owner.mutation, 16.)
    assert sorted(i.fitness.value for i in controller.refinement_parents(2)) == [1., 3.]
