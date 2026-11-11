"""TRAIN-only procedure shared by fixed and staged mutation schedules."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

from benchmark.industrial.experiments.mutation_diversity.study import MutationStudy, AuditedPreparation


def load_training_protocol():
    value = json.loads(Path(__file__).with_name('training.json').read_text('utf-8'))
    if value['schema_version'] != 'industrial_staged_training@1' or value['training_enabled'] is not True:
        raise ValueError('Unsupported training protocol')
    for key in ('budget_seconds', 'candidate_seconds', 'workers', 'native_threads', 'minimum_generations',
                'full_process_seconds', 'pilot_seconds', 'pilot_candidate_seconds', 'population_size',
                'max_candidates', 'max_nodes', 'max_depth', 'max_arity', 'forecast_outer_horizon'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if value['workers'] != 4 or value['native_threads'] != 1 or value['full_process_seconds'] > 1200:
        raise ValueError('Execution exceeds the approved configuration')
    if value['seeds'] != [42, 123, 2026] or set(value['tasks']) != {'classification', 'regression', 'ts_forecasting'}:
        raise ValueError('Unexpected comparison matrix')
    if not 0 < value['classification_outer_fraction'] < 1 or not 0 < value['family_fraction'] <= 1:
        raise ValueError('Invalid holdout or family fraction')
    return value


def search_policy(mode):
    from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
    protocol = load_training_protocol()
    policy = load_staged_search_defaults()
    stages = tuple(replace(stage, family_fraction=protocol['family_fraction']) for stage in policy.stages)
    if mode == 'fixed':
        stages = (replace(stages[1], name='fixed', until=1.),)
    elif mode != 'staged':
        raise ValueError('mode must be fixed or staged')
    return replace(policy, stages=stages)


@dataclass(frozen=True)
class StagedTrainingStudy(MutationStudy):
    pilot: bool = False

    @property
    def strategy(self):
        from fedot_ind.core.optimizer.configuration import MutationStrategy
        return MutationStrategy.STRUCTURAL_BALANCED

    def load_data(self):
        from benchmark.industrial.experiments.mutation_diversity.data import load_training_data
        return load_training_data(self.task, load_training_protocol())

    def implementation_paths(self):
        return [*Path(__file__).parent.glob('*.py'), Path(__file__).with_name('training.json'),
                Path('fedot_ind/core/optimizer/search_policy/defaults.json'),
                *Path('fedot_ind/core/operation/transformation/basis').glob('*.py'),
                *Path('fedot_ind/core/models/ts_forecasting/branching').glob('*.py'),
                *Path('fedot_ind/integration/fedot').glob('*.py'),
                *Path('fedot_ind/integration/fedot/extensions').glob('*.py'),
                Path('fedot_ind/integration/fedot/extensions/catalog.json'),
                Path('benchmark/industrial/experiments/mutation_diversity/study.py'),
                Path('benchmark/industrial/experiments/mutation_diversity/data.py')]

    def search_settings(self, settings):
        protocol = load_training_protocol()
        result = deepcopy(settings)
        for key in ('tasks', 'operation_parameters', 'population_size', 'max_depth', 'max_arity'):
            result[key] = deepcopy(protocol[key])
        result['candidate_seconds'] = protocol['pilot_candidate_seconds' if self.pilot else 'candidate_seconds']
        result['budget_seconds'] = protocol['pilot_seconds' if self.pilot else 'budget_seconds']
        return result

    def configure_environment(self, params):
        if self.task == 'ts_forecasting':
            from fedot.core.pipelines.verification import verifier_for_task
            from fedot.core.repository.tasks import TaskTypesEnum
            from fedot_ind.integration.fedot.forecast_branches import ForecastPipelineAdapter
            from fedot_ind.core.optimizer.graph_validation import IndustrialGraphVerifier
            params.adapter = ForecastPipelineAdapter()
            params.verifier = IndustrialGraphVerifier(
                adapter=params.adapter, task_type=self.task,
                legacy_verifier=verifier_for_task(TaskTypesEnum.ts_forecasting, params.adapter))
        return params

    def configure_evolution(self, config):
        from fedot_ind.core.optimizer.configuration import MutationAgentType, MutationPlanningPolicy
        protocol = load_training_protocol()
        return replace(config, mutation_schedule=search_policy(self.arm), mutation_agent=MutationAgentType.RANDOM,
                       mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=protocol['max_candidates'],
                                                                max_nodes=protocol['max_nodes']))

    def prepare_objective(self, measured, data):
        from fedot_ind.integration.fedot.evaluation import ConfirmedObjective
        return ConfirmedObjective.from_training_data(measured, tuple(train for train, validation in data.folds))

    def preparation(self, base, root):
        from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
        preparation = ParameterSearchPreparation.from_defaults(
            base.adapter, json.loads(base.parameters_json), search_policy(self.arm).parameters, self.task)
        return AuditedPreparation(preparation, str(root / 'preparation.jsonl'))

    def configure_selected(self, pipeline, parameters):
        # Every node has already been prepared before evaluation. Do not reset its chosen grid values.
        return pipeline

    def evaluate_selected(self, winner, data, root):
        if not self.pilot:
            return super().evaluate_selected(winner, data, root)
        from time import perf_counter
        import numpy as np
        from benchmark.industrial.experiments.evolution_parallelism.final_evaluation import FinalEvaluation
        from benchmark.industrial.experiments.evolution_parallelism.search import to_numpy
        train, validation = data.folds[0]
        try:
            start = perf_counter()
            winner.fit(train)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            output = winner.predict(replace(validation, target=None, state='predict'),
                                    output_mode='labels' if self.task == 'classification' else 'default')
            prediction_seconds = perf_counter() - start
            if not np.isfinite(to_numpy(output.predict)).all():
                raise ValueError('Pilot returned nonfinite predictions')
            return FinalEvaluation(fit_seconds, prediction_seconds, None), None
        finally:
            winner.unfit()
