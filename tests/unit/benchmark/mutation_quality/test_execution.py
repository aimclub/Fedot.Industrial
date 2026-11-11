from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmark.industrial.experiments.mutation_quality.arms import policy_for_arm
from benchmark.industrial.experiments.mutation_quality.config import Arm


def test_all_arms_share_models_parameters_and_family_reservation():
    from benchmark.industrial.experiments.mutation_quality.study import QualityStudy, arm_policy
    from fedot_ind.core.optimizer.configuration import EvolutionConfig
    studies = [QualityStudy(arm.value, 'regression', scenario_name='NaturalGasPricesSentiment') for arm in Arm]
    configs = [study.configure_evolution(EvolutionConfig()) for study in studies]
    assert len({repr(replace(config, mutation_schedule=None)) for config in configs}) == 1
    settings = [study.search_settings({}) for study in studies]
    assert settings[0] == settings[1] == settings[2]
    policies = [arm_policy(arm) for arm in Arm]
    assert policies[0].parameters == policies[1].parameters == policies[2].parameters
    assert {stage.family_fraction for policy in policies for stage in policy.stages} == {.5}
    assert policies[0].stages[0].weights.rewrite == 0
    assert policies[0].stages[0].weights.parameter_change == pytest.approx(.625)
    assert [stage.until for stage in policies[2].stages] == [.3, .7, 1.]


@pytest.mark.parametrize('arm', list(Arm))
def test_configured_actions_exclude_rewrite_entirely_for_baseline(arm):
    from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
    from fedot_ind.core.optimizer.search_policy.runtime import configure_actions
    from fedot_ind.core.optimizer.search_policy.parameters import rewrite_subgraph_mutation
    from fedot_ind.core.optimizer.mutation import mutation_name
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from benchmark.industrial.experiments.mutation_quality.study import arm_policy
    original = SimpleNamespace(mutation_types=[MutationTypesEnum.single_change, MutationTypesEnum.single_add,
                                               MutationTypesEnum.single_drop, rewrite_subgraph_mutation])
    snapshot = deepcopy(original.mutation_types)
    configured = configure_actions(original, arm_policy(arm))
    names = list(map(mutation_name, configured.mutation_types))
    assert original.mutation_types == snapshot
    assert len(names) == len(set(names))
    assert ('rewrite_subgraph_mutation' in names) == (arm is not Arm.BASELINE)
    assert policy_for_arm(arm, load_staged_search_defaults(), .5) == arm_policy(arm)


def test_baseline_runtime_never_enumerates_rewrites_without_training():
    from benchmark.industrial.experiments.mutation_planning.runner import _environment
    from benchmark.industrial.experiments.mutation_quality.study import arm_policy
    from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationPlanningPolicy
    from fedot_ind.core.optimizer.mutation_planning import RawInputSpec
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from fedot.core.operations.operation import Operation
    from golem.core.optimisers.objective import Objective
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
    from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
    from golem.core.optimisers.opt_history_objects.individual import Individual
    requirements, params, graph = _environment(dict(name='quality_baseline', task='regression',
                                                    operations=['scaling', 'normalization', 'ridge'], initial_chain=['scaling', 'ridge']),
                                               dict(max_depth=6, max_arity=4))
    requirements.timeout = timedelta(seconds=30)
    algorithm = GPAlgorithmParameters(mutation_prob=1., mutation_types=[MutationTypesEnum.single_change,
                                                                        MutationTypesEnum.single_add, MutationTypesEnum.single_drop])
    optimizer = IndustrialEvoOptimizer(Objective({'loss': lambda g: 0.}), [graph], requirements, params, algorithm,
                                       EvolutionConfig(mutation_planning=MutationPlanningPolicy(enabled=True), mutation_schedule=arm_policy('baseline')))
    optimizer.mutation.raw_input = RawInputSpec('tabular', True)
    records = []
    optimizer.mutation.observer = lambda event: records.append(event.to_record())
    with patch.object(Operation, 'fit', side_effect=AssertionError('No training in structural test')) as fit:
        with operator_random_scope(42):
            for _ in range(10):
                optimizer.mutation._mutate(Individual(deepcopy(graph)))
        assert fit.call_count == 0
    assert len(records) == 10
    for row in records:
        assert row['action'] != 'rewrite_subgraph_mutation'
        assert 'rewrite' not in row['planning']['enabled_kinds']
        assert not set(row['planning']['family_counts']) & {'rewrite_chain', 'fork_feature_branch',
                                                            'fork_forecast_branch', 'prune_feature_branch'}


def test_outer_values_do_not_affect_inner_normalization():
    from benchmark.industrial.experiments.mutation_quality.data import normalized_pair
    x = np.arange(60, dtype=np.float32).reshape(10, 2, 3)
    target = np.arange(10)
    expected = normalized_pair(x, target, [0, 1, 2], [3, 4])
    x[5:] = 1e9
    actual = normalized_pair(x, target, [0, 1, 2], [3, 4])
    for (left, ly), (right, ry) in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(ly, ry)


def test_preview_never_starts_training(tmp_path, monkeypatch, capsys):
    from benchmark.industrial.experiments.mutation_quality import runner
    from benchmark.industrial.experiments.mutation_quality.__main__ import main
    monkeypatch.setattr(runner, 'run_technical', lambda *a: pytest.fail('Unexpected training'))
    root = tmp_path / 'unused'
    monkeypatch.setattr(sys, 'argv', ['mutation_quality', '--technical', '--output-dir', str(root)])
    main()
    assert json.loads(capsys.readouterr().out)['training_started'] is False
    assert not root.exists()
    assert len(runner.technical_jobs()) == len(set(runner.technical_jobs())) == 9


def test_unknown_worker_request_rejected_before_effects(tmp_path):
    from benchmark.industrial.experiments.mutation_quality.runner import run_trial
    with pytest.raises(ValueError, match='bounded technical matrix'):
        run_trial(tmp_path / 'unused', 'Tecator', 'baseline', 42)
    assert not (tmp_path / 'unused').exists()
