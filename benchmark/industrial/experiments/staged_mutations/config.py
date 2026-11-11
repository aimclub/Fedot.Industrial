"""Both future arms share operators, grids, limits and family reservation."""
from dataclasses import replace
import json
from pathlib import Path

from fedot_ind.core.optimizer.configuration import (EvolutionConfig, ExecutionPolicy, MutationPlanningPolicy)
from fedot_ind.core.optimizer.evaluation.configuration import EvaluationBackend, EvaluationExecutionConfig
from fedot_ind.core.optimizer.evaluation.dask_config import DaskExecutionConfig, DaskLifecycle
from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults


def load_protocol():
    result = json.loads(Path(__file__).with_name('protocol.json').read_text('utf-8'))
    if result['schema_version'] != 'industrial_staged_mutation_comparison@1' or result['training_enabled'] is not False:
        raise ValueError('Only the no-training staged search protocol is implemented here')
    return result


def build_config(mode='staged', seed=42):
    protocol = load_protocol()
    policy = load_staged_search_defaults()
    stages = tuple(replace(stage, family_fraction=protocol['family_fraction']) for stage in policy.stages)
    if mode == 'fixed':
        stages = (replace(stages[1], name='fixed', until=1.),)
    elif mode != 'staged':
        raise ValueError('Mode must be fixed or staged')
    return EvolutionConfig(random_seed=seed, mutation_schedule=replace(policy, stages=stages),
                           mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=protocol['max_candidates'],
                                                                    max_nodes=protocol['max_nodes']),
                           execution_policy=ExecutionPolicy(retry_initial_population_without_timer=False),
                           evaluation_execution=EvaluationExecutionConfig(EvaluationBackend.DASK,
                                                                          total_timeout_seconds=protocol[
                                                                              'budget_seconds'], candidate_timeout_seconds=protocol['candidate_seconds'],
                                                                          dask=DaskExecutionConfig(lifecycle=DaskLifecycle.SEARCH)))
