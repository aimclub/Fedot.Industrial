"""An explicit cyclic policy, separate from mixed structural stages."""
from dataclasses import dataclass

from .contracts import MutationMix, ParameterGrid, StagedSearchPolicy, number


@dataclass(frozen=True)
class ParameterOnlyWeights:
    def to_record(self):
        return dict(parameter_change=1., replace=0., add=0., drop=0., rewrite=0.)


@dataclass(frozen=True)
class CyclicStage:
    name: str
    until: float
    weights: MutationMix | ParameterOnlyWeights
    family_fraction: float
    conservative: bool


@dataclass(frozen=True)
class CyclicSearchPolicy:
    parameters: tuple[ParameterGrid, ...]
    exploration: MutationMix
    cycles: int = 3
    active_fraction: float = .5
    family_fraction: float = .5
    max_families: int = 16
    batch_size: int = 4

    def __post_init__(self):
        if not isinstance(self.exploration, MutationMix) or self.exploration.parameter_change != 0:
            raise ValueError('Active phases must use only structural actions')
        for key in ('cycles', 'batch_size'):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= 32:
                raise ValueError(f'{key} must be an integer from one to 32')
        number(self.active_fraction, 'active_fraction', maximum=1.)
        if self.active_fraction in (0., 1.):
            raise ValueError('Each cycle must reserve both phases')
        number(self.family_fraction, 'family_fraction', maximum=1.)
        if self.family_fraction <= 0:
            raise ValueError('Family reservation must be positive')
        # Reuse the existing grid and bounded archive contract, not its mixed-phase semantics.
        self._check_grids()

    def _check_grids(self):
        from .contracts import SearchStage
        StagedSearchPolicy((SearchStage('validation', 1., self.exploration, self.family_fraction),),
                           self.parameters, self.max_families)

    @property
    def stages(self):
        return tuple(stage for cycle in range(self.cycles) for stage in (
            CyclicStage(f'cycle_{cycle + 1}_active', (cycle + self.active_fraction) / self.cycles,
                        self.exploration, self.family_fraction, False),
            CyclicStage(f'cycle_{cycle + 1}_refine', (cycle + 1) / self.cycles,
                        ParameterOnlyWeights(), self.family_fraction, True)))

    def to_record(self):
        return dict(schema_version='industrial_cyclic_search@1', cycles=self.cycles,
                    active_fraction=self.active_fraction, family_fraction=self.family_fraction,
                    max_families=self.max_families, batch_size=self.batch_size,
                    exploration=self.exploration.to_record(), parameters=[p.to_record() for p in self.parameters])


def parse_cyclic_search(payload):
    if (type(payload) is not dict or set(payload) != {'schema_version', 'cycles', 'active_fraction',
                                                      'family_fraction', 'max_families', 'batch_size', 'exploration', 'parameters'}
            or payload['schema_version'] != 'industrial_cyclic_search@1'):
        raise ValueError('Invalid cyclic policy shape')
    weights = payload['exploration']
    if type(weights) is not dict or set(weights) != {'parameter_change', 'replace', 'add', 'drop', 'rewrite'}:
        raise ValueError('Invalid exploration weights')
    if type(payload['parameters']) is not list:
        raise ValueError('Cyclic parameters must be an array')
    grids = []
    for row in payload['parameters']:
        if type(row) is not dict or set(row) != {'operation', 'parameter', 'values'} or type(row['values']) is not list:
            raise ValueError('Invalid cyclic parameter grid')
        grids.append(ParameterGrid(row['operation'], row['parameter'], tuple(row['values'])))
    return CyclicSearchPolicy(tuple(grids), MutationMix(**weights), payload['cycles'], payload['active_fraction'],
                              payload['family_fraction'], payload['max_families'], payload['batch_size'])
