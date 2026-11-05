"""B and D differ in ordering, not data, parameter domains or evaluation batching."""
from dataclasses import dataclass
import json
from pathlib import Path

from benchmark.industrial.experiments.mutation_quality.execution.study import FrozenStudy
from benchmark.industrial.experiments.mutation_diversity.study import AuditedPreparation
from fedot_ind.core.optimizer.search_policy import MutationMix, SearchStage, StagedSearchPolicy
from fedot_ind.core.optimizer.search_policy.cyclic import CyclicSearchPolicy

from .profile import parameter_profile, validate_training_bounds


def policy_for(arm):
    value = json.loads(Path(__file__).with_name('schedule.json').read_text('utf-8'))
    if value['schema_version'] != 'industrial_bd_schedule@1':
        raise ValueError('Invalid B/D schedule')
    grids = parameter_profile()
    if arm == 'fixed':
        return StagedSearchPolicy((SearchStage('mixed', 1., MutationMix(**value['mixed']), value['family_fraction']),),
                                  grids, value['max_families'])
    if arm != 'cyclic':
        raise ValueError('B/D arm must be fixed or cyclic')
    return CyclicSearchPolicy(grids, MutationMix(**value['exploration']), value['cycles'], value['active_fraction'],
                              value['family_fraction'], value['max_families'], value['batch_size'])


@dataclass(frozen=True)
class GeneratorStudy(FrozenStudy):
    keep_history = True

    @property
    def policy(self):
        return policy_for(self.arm)

    def load_data(self):
        data = super().load_data()
        validate_training_bounds(data, self.policy.parameters)
        return data

    def preparation(self, base, root):
        from fedot_ind.core.optimizer.search_policy.preparation import ParameterSearchPreparation
        return AuditedPreparation(ParameterSearchPreparation.from_defaults(
            base.adapter, json.loads(base.parameters_json), self.policy.parameters, self.task, normalize_generators=True),
            str(root / 'preparation.jsonl'))

    def observe_mutations(self, optimizer, root):
        from fedot_ind.core.optimizer.search_policy.batches import install_bounded_reproduction
        super().observe_mutations(optimizer, root)
        value = json.loads(Path(__file__).with_name('schedule.json').read_text('utf-8'))
        install_bounded_reproduction(optimizer, value['batch_size'])

    def implementation_paths(self):
        return [*super().implementation_paths(), *Path(__file__).parent.glob('*.py'),
                *Path(__file__).parent.glob('*.json'), *Path('fedot_ind/core/optimizer/search_policy').glob('*.py'),
                *Path('fedot_ind/core/tuning').glob('*.json'), Path('fedot_ind/core/tuning/generator_catalog.py')]
