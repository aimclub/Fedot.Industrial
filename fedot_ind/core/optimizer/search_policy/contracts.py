"""Validated opt-in policy for time-budgeted structure and parameter search."""
from dataclasses import dataclass
from math import isfinite


def number(value, name, minimum=0., maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside the allowed range')


@dataclass(frozen=True)
class ParameterGrid:
    operation: str
    parameter: str
    values: tuple

    def __post_init__(self):
        for value in (self.operation, self.parameter):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Operation and parameter names must be non-empty strings')
        protected = {'task_type', 'forecast_length', 'random_state', 'random_seed', 'seed', 'n_jobs',
                     'num_threads', 'device', 'use_cache'}
        if self.parameter in protected:
            raise ValueError(f'Execution or context parameter {self.parameter} cannot be searched')
        if not isinstance(self.values, tuple) or not 1 <= len(self.values) <= 32:
            raise ValueError('A parameter grid must contain one to 32 values')
        seen = set()
        for value in self.values:
            if type(value) not in (str, bool, int, float, type(None)):
                raise ValueError('Parameter choices must be JSON scalars')
            if type(value) is float and not isfinite(value):
                raise ValueError('Parameter choices must be finite')
            identity = (type(value), value)
            if identity in seen:
                raise ValueError('Parameter choices must be distinct')
            seen.add(identity)

    def to_record(self):
        return dict(operation=self.operation, parameter=self.parameter, values=list(self.values))


@dataclass(frozen=True)
class MutationMix:
    parameter_change: float
    replace: float
    add: float
    drop: float
    rewrite: float

    def __post_init__(self):
        for name, value in self.to_record().items():
            number(value, name, maximum=1.)
        if abs(sum(self.to_record().values()) - 1.) > 1e-9:
            raise ValueError('Mutation weights must sum to one')
        if self.parameter_change == 1.:
            raise ValueError('Each stage must retain structural exploration')

    def to_record(self):
        return dict(parameter_change=self.parameter_change, replace=self.replace,
                    add=self.add, drop=self.drop, rewrite=self.rewrite)


@dataclass(frozen=True)
class SearchStage:
    name: str
    until: float
    weights: MutationMix
    family_fraction: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('A stage requires a name')
        number(self.until, 'until', maximum=1.)
        number(self.family_fraction, 'family_fraction', maximum=1.)
        if self.until == 0 or not isinstance(self.weights, MutationMix):
            raise ValueError('A stage requires a positive boundary and typed weights')

    def to_record(self):
        return dict(name=self.name, until=self.until, weights=self.weights.to_record(),
                    family_fraction=self.family_fraction)


@dataclass(frozen=True)
class StagedSearchPolicy:
    stages: tuple[SearchStage, ...]
    parameters: tuple[ParameterGrid, ...]
    max_families: int = 16

    def __post_init__(self):
        if not isinstance(self.stages, tuple) or not self.stages or any(
                not isinstance(stage, SearchStage) for stage in self.stages):
            raise ValueError('Stages must be a non-empty tuple of SearchStage values')
        if self.stages[-1].until != 1 or any(a.until >= b.until for a, b in zip(self.stages, self.stages[1:])):
            raise ValueError('Stage boundaries must increase and end at one')
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError('Stage names must be distinct')
        if not isinstance(self.parameters, tuple) or not self.parameters or any(
                not isinstance(grid, ParameterGrid) for grid in self.parameters):
            raise ValueError('A staged search requires explicit parameter grids')
        if len({(grid.operation, grid.parameter) for grid in self.parameters}) != len(self.parameters):
            raise ValueError('Parameter grids must not overlap')
        if isinstance(self.max_families, bool) or not isinstance(self.max_families, int) or self.max_families < 1:
            raise ValueError('max_families must be a positive integer')

    def to_record(self):
        return dict(schema_version='industrial_staged_search@1', stages=[s.to_record() for s in self.stages],
                    parameters=[p.to_record() for p in self.parameters], max_families=self.max_families)


def parse_staged_search(payload):
    def shape(value, keys):
        if type(value) is not dict or set(value) != set(keys):
            raise ValueError(f'Expected exactly these fields: {tuple(keys)}')
    shape(payload, ('schema_version', 'stages', 'parameters', 'max_families'))
    if payload['schema_version'] != 'industrial_staged_search@1':
        raise ValueError('Unsupported staged search schema')
    if type(payload['stages']) is not list or type(payload['parameters']) is not list:
        raise ValueError('Stages and parameter grids must be arrays')
    stages, parameters = [], []
    for item in payload['stages']:
        shape(item, ('name', 'until', 'weights', 'family_fraction'))
        shape(item['weights'], ('parameter_change', 'replace', 'add', 'drop', 'rewrite'))
        stages.append(SearchStage(item['name'], item['until'], MutationMix(**item['weights']), item['family_fraction']))
    for item in payload['parameters']:
        shape(item, ('operation', 'parameter', 'values'))
        if type(item['values']) is not list:
            raise ValueError('Grid values must be an array')
        parameters.append(ParameterGrid(item['operation'], item['parameter'], tuple(item['values'])))
    return StagedSearchPolicy(tuple(stages), tuple(parameters), payload['max_families'])
