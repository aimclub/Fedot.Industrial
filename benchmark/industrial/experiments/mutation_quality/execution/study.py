"""Effect adapter using only sealed settings; full and smoke share model logic."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

from benchmark.industrial.experiments.staged_mutations.training import StagedTrainingStudy
from benchmark.industrial.experiments.mutation_diversity.study import MutationStudy, AuditedPreparation

from .frozen_data import load_frozen_data, load_control


@dataclass(frozen=True)
class FrozenStudy(StagedTrainingStudy):
    bundle: str = ''
    lock: object = None
    trial: object = None

    @property
    def policy(self):
        from fedot_ind.core.optimizer.search_policy.contracts import parse_staged_search
        return parse_staged_search(deepcopy(self.lock['policies'][self.arm]))

    def load_data(self):
        return load_frozen_data(self.bundle, self.lock, self.trial.scenario)

    def implementation_paths(self):
        return [*(Path(path) for path in self.lock['sources']), *Path(__file__).parent.glob('*.py'),
                Path(__file__).with_name('launch.json')]

    def search_settings(self, settings):
        training = self.lock['training_reference']
        result = deepcopy(settings)
        for key in ('operation_parameters', 'population_size', 'max_depth', 'max_arity'):
            result[key] = deepcopy(training[key])
        result['tasks'] = {self.task: deepcopy(self.lock['scenario_settings'][self.trial.scenario])}
        result['budget_seconds'] = self.trial.seconds
        result['candidate_seconds'] = self.lock['protocol']['candidate_seconds']
        return result

    def configure_evolution(self, config):
        from fedot_ind.core.optimizer.configuration import MutationAgentType, MutationPlanningPolicy
        training = self.lock['training_reference']
        return replace(config, mutation_schedule=self.policy, mutation_agent=MutationAgentType.RANDOM,
                       mutation_planning=MutationPlanningPolicy(enabled=True, max_candidates=training['max_candidates'],
                                                                max_nodes=training['max_nodes']))

    def preparation(self, base, root):
        from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
        return AuditedPreparation(ParameterSearchPreparation.from_defaults(
            base.adapter, json.loads(base.parameters_json), self.policy.parameters, self.task),
            str(root / 'preparation.jsonl'))

    def observe_mutations(self, optimizer, root):
        from fedot_ind.core.optimizer.randomness import SeededOperator
        from fedot_ind.core.optimizer.mutation import mutation_name
        mutation = optimizer.mutation
        while isinstance(mutation, SeededOperator):
            mutation = mutation.operator
        names = list(map(mutation_name, mutation.operator.agent.available_actions))
        if ('rewrite_subgraph_mutation' in names) != (self.arm != 'baseline'):
            raise AssertionError('Actual mutation actions violate the frozen arm')
        (root / 'arm_contract.json').write_text(json.dumps(dict(arm=self.arm, actions=names,
                                                                policy=self.policy.to_record()), indent=2) + '\n', encoding='utf-8')
        MutationStudy.observe_mutations(self, optimizer, root)

    def evaluate_selected(self, winner, data, root):
        if self.pilot:
            return StagedTrainingStudy.evaluate_selected(self, winner, data, root)
        # First access to control occurs only after the winning configuration is fixed.
        control = load_control(self.bundle, self.lock, self.trial.scenario)
        return MutationStudy.evaluate_selected(self, winner, replace(data, test=control), root)
