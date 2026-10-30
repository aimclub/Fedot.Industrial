"""A/B/C adapter for the existing evolution interpreter; technical runs only."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

from benchmark.industrial.experiments.staged_mutations.training import StagedTrainingStudy, load_training_protocol

from .arms import policy_for_arm
from .config import Arm, load_protocol


def arm_policy(arm):
    from fedot_ind.core.optimizer.search_policy import load_staged_search_defaults
    return policy_for_arm(arm, load_staged_search_defaults(), load_training_protocol()['family_fraction'])


@dataclass(frozen=True)
class QualityStudy(StagedTrainingStudy):
    pilot: bool = True
    scenario_name: str = 'ECG200'

    def __post_init__(self):
        Arm(self.arm)
        if self.pilot is not True:
            raise ValueError('Only explicit technical trials are enabled')
        if self.scenario.task != self.task:
            raise ValueError('Scenario and task do not match')

    @property
    def scenario(self):
        matches = [row for row in load_protocol().scenarios if row.name == self.scenario_name]
        if not matches:
            raise ValueError('Unknown scenario')
        return matches[0]

    def load_data(self):
        from .data import load_search_data
        return load_search_data(self.scenario, load_protocol(), Path(__file__).resolve().parents[4])

    def implementation_paths(self):
        return [*super().implementation_paths(), *Path(__file__).parent.glob('*.py'),
                *Path(__file__).parent.glob('*.json'), Path('fedot_ind/tools/ts_parser.py')]

    def search_settings(self, settings):
        result = super().search_settings(settings)
        technical = json.loads(Path(__file__).with_name('technical.json').read_text('utf-8'))
        result['budget_seconds'] = technical['search_seconds']
        result['candidate_seconds'] = load_protocol().candidate_seconds
        result['tasks'] = deepcopy(result['tasks'])
        result['tasks'][self.task]['dataset'] = self.scenario_name
        return result

    def configure_evolution(self, config):
        from fedot_ind.core.optimizer.configuration import MutationAgentType, MutationPlanningPolicy
        training = load_training_protocol()
        return replace(config, mutation_schedule=arm_policy(self.arm), mutation_agent=MutationAgentType.RANDOM,
                       mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=training['max_candidates'],
                                                                max_nodes=training['max_nodes']))

    def preparation(self, base, root):
        from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
        from benchmark.industrial.experiments.mutation_diversity.study import AuditedPreparation
        preparation = ParameterSearchPreparation.from_defaults(
            base.adapter, json.loads(base.parameters_json), arm_policy(self.arm).parameters, self.task)
        return AuditedPreparation(preparation, str(root / 'preparation.jsonl'))

    def observe_mutations(self, optimizer, root):
        from fedot_ind.core.optimizer.randomness import SeededOperator
        from fedot_ind.core.optimizer.mutation import mutation_name
        mutation = optimizer.mutation
        while isinstance(mutation, SeededOperator):
            mutation = mutation.operator
        names = tuple(map(mutation_name, mutation.operator.agent.available_actions))
        expected_rewrite = Arm(self.arm) is not Arm.BASELINE
        if ('rewrite_subgraph_mutation' in names) != expected_rewrite:
            raise AssertionError(f'Actual runtime actions violate arm contract: {names}')
        (root / 'arm_contract.json').write_text(json.dumps(dict(
            arm=self.arm, actions=names, rewrite_enabled=expected_rewrite,
            policy=arm_policy(self.arm).to_record()), indent=2) + '\n', encoding='utf-8')
        super().observe_mutations(optimizer, root)
