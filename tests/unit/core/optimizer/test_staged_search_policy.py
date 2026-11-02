import json
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.optimizer.configuration import (
    EvolutionConfig, EvolutionConfigError, MutationPlanningPolicy, normalize_evolution_config)
from fedot_ind.core.optimizer.search_policy import ParameterGrid, parse_staged_search, stage_for_budget
from fedot_ind.core.optimizer.search_policy.planning import FamilyCandidate, select_survivors


def policy():
    import fedot_ind.core.optimizer.search_policy as module
    return parse_staged_search(json.loads(Path(module.__file__).with_name('defaults.json').read_text('utf-8')))


def test_default_search_is_unchanged_and_new_transport_round_trips():
    assert EvolutionConfig().mutation_schedule is None
    assert 'mutation_schedule' not in EvolutionConfig().to_record()
    config = EvolutionConfig(mutation_planning=MutationPlanningPolicy(enabled=True), mutation_schedule=policy())
    assert normalize_evolution_config(json.loads(json.dumps(config.to_record()))) == config


@pytest.mark.parametrize('elapsed,name', [(0, 'explore'), (29, 'explore'), (30, 'joint'),
                                          (70, 'refine'), (100, 'refine'), (200, 'refine')])
def test_stages_follow_elapsed_budget_not_generation_count(elapsed, name):
    assert stage_for_budget(policy(), elapsed, 100).name == name


@given(st.floats(min_value=0, max_value=10000, allow_nan=False),
       st.floats(min_value=0, max_value=10000, allow_nan=False))
def test_stage_order_cannot_go_back_when_elapsed_time_increases(a, b):
    p = policy()
    first, last = sorted((a, b))
    assert p.stages.index(stage_for_budget(p, first, 100)) <= p.stages.index(stage_for_budget(p, last, 100))


@pytest.mark.parametrize('name', ['random_state', 'task_type', 'forecast_length', 'n_jobs', 'device'])
def test_context_and_execution_parameters_cannot_be_searched(name):
    with pytest.raises(ValueError):
        ParameterGrid('model', name, (1, 2))


@pytest.mark.parametrize('modify', [lambda x: x.update(unknown=True),
                                    lambda x: x['stages'][0].update(
                                        until=True), lambda x: x['stages'][-1].update(until=.9),
                                    lambda x: x['stages'][0]['weights'].update(rewrite=float('nan')),
                                    lambda x: x['parameters'][0].update(values=[{}]), lambda x: x.update(max_families=False)])
def test_invalid_policy_is_rejected_at_the_boundary(modify):
    payload = policy().to_record()
    modify(payload)
    with pytest.raises(ValueError):
        parse_staged_search(payload)
    with pytest.raises(EvolutionConfigError):
        normalize_evolution_config({'mutation_schedule': payload, 'mutation_planning': {'enabled': True}})


def test_schedule_requires_planned_mutation_and_random_agent():
    with pytest.raises(EvolutionConfigError):
        normalize_evolution_config({'mutation_schedule': policy().to_record()})


def test_legacy_optimizer_does_not_silently_ignore_the_new_policy():
    from types import SimpleNamespace
    from fedot_ind.core.optimizer.FedotEvoOptimizer import FedotEvoOptimizer
    optimizer = object.__new__(FedotEvoOptimizer)
    config = EvolutionConfig(mutation_planning=MutationPlanningPolicy(enabled=True), mutation_schedule=policy())
    with pytest.raises(ValueError, match='IndustrialEvoOptimizer'):
        optimizer._set_optimisation_strategy(SimpleNamespace(), config)


def test_family_slots_keep_the_global_best_and_a_distinct_family():
    candidates = (FamilyCandidate('a1', 'a', 1., 3), FamilyCandidate('a2', 'a', 1.1, 3),
                  FamilyCandidate('b1', 'b', 2., 2), FamilyCandidate('c1', 'c', 3., 2))
    assert select_survivors(candidates, 2, 1.) == ('a1', 'b1')
    assert select_survivors(candidates, 2, 0.) == ('a1', 'a2')
    assert select_survivors(tuple(reversed(candidates)), 2, 1.) == ('a1', 'b1')


def test_bounded_parameter_callback_detaches_and_changes_only_a_declared_value():
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot_ind.core.optimizer.search_policy.parameters import BoundedParameterMutation
    pipeline = PipelineBuilder().add_node('ridge', params={'alpha': 1., 'solver': 'lsqr'}).build()
    mutation = BoundedParameterMutation((ParameterGrid('ridge', 'alpha', (.1, 1., 10.)),))
    result = mutation(pipeline)
    assert result is not pipeline
    assert pipeline.root_node.parameters == {'alpha': 1., 'solver': 'lsqr'}
    assert result.root_node.parameters['alpha'] in (.1, 10.)
    assert result.root_node.parameters['solver'] == 'lsqr'
