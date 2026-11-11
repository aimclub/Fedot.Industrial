"""Typed preparation contract; it does not authorize or launch training."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path, PurePosixPath


class Arm(str, Enum):
    BASELINE = 'baseline'
    FIXED = 'expanded_fixed'
    STAGED = 'expanded_staged'


@dataclass(frozen=True)
class Scenario:
    name: str
    task: str
    source: str
    path: str | None
    group: str


@dataclass(frozen=True)
class QualityProtocol:
    scenarios: tuple[Scenario, ...]
    seeds: tuple[int, ...]
    arms: tuple[Arm, ...]
    split_seed: int
    outer_fraction: float
    inner_folds: int
    budget_seconds: int
    candidate_seconds: int
    workers: int
    native_threads: int
    minimum_generations: int
    forecast_horizon: int
    minimum_series_length: int
    training_reference: str
    policy_reference: str

    def jobs(self):
        # Rotate arm order without consuming model/search RNG.
        return tuple((scenario.name, self.arms[(dataset_index + seed_index + offset) % len(self.arms)].value, seed)
                     for dataset_index, scenario in enumerate(self.scenarios)
                     for seed_index, seed in enumerate(self.seeds)
                     for offset in range(len(self.arms)))


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a repository-relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Path must stay inside the repository')
    return value


def parse_protocol(value):
    fields = set(QualityProtocol.__dataclass_fields__) | {'schema_version', 'training_enabled'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('Unexpected protocol fields')
    if value['schema_version'] != 'industrial_mutation_quality@1' or value['training_enabled'] is not False:
        raise ValueError('Only preparation without training is supported')
    for key in ('split_seed', 'inner_folds', 'budget_seconds', 'candidate_seconds', 'workers',
                'native_threads', 'minimum_generations', 'forecast_horizon', 'minimum_series_length'):
        if type(value[key]) is not int or value[key] < (0 if key == 'split_seed' else 1):
            raise ValueError(f'{key} must be an integer in range')
    if value['inner_folds'] < 2 or value['minimum_series_length'] <= 4 * value['forecast_horizon']:
        raise ValueError('Insufficient folds or temporal history')
    if (type(value['outer_fraction']) not in (int, float) or not 0 < value['outer_fraction'] < 1
            or value['workers'] != 4 or value['native_threads'] != 1
            or value['candidate_seconds'] > value['budget_seconds']):
        raise ValueError('Invalid holdout or approved execution limits')
    seeds = value['seeds']
    if (type(seeds) is not list or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if value['arms'] != [arm.value for arm in Arm]:
        raise ValueError('Require the three ordered comparison arms')
    scenarios = []
    if type(value['scenarios']) is not list or not value['scenarios']:
        raise ValueError('Require explicit scenarios')
    for row in value['scenarios']:
        if type(row) is not dict or set(row) != set(Scenario.__dataclass_fields__):
            raise ValueError('Unexpected scenario fields')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ('name', 'task', 'source', 'group')):
            raise ValueError('Scenario fields must be nonempty strings')
        if row['task'] not in ('classification', 'regression', 'ts_forecasting'):
            raise ValueError('Unsupported task')
        allowed = {'classification': ('ts', 'tsv'), 'regression': ('ts', 'tsv', 'sktime_tecator'),
                   'ts_forecasting': ('airline_snapshot', 'wide_csv', 'monash_train_csv', 'pending_external')}
        if row['source'] not in allowed[row['task']]:
            raise ValueError('Source does not match task')
        if row['source'] in ('sktime_tecator', 'pending_external'):
            if row['path'] is not None:
                raise ValueError('This source must not declare a local path')
        else:
            relative_path(row['path'])
            if row['source'] != 'airline_snapshot' and '_TRAIN.' not in row['path'] and '-train.' not in row['path']:
                raise ValueError('Only an explicit TRAIN file is allowed')
        scenarios.append(Scenario(**row))
    if len({row.name for row in scenarios}) != len(scenarios):
        raise ValueError('Scenario names must be unique')
    return QualityProtocol(tuple(scenarios), tuple(seeds), tuple(Arm), **{
        key: relative_path(value[key]) if key.endswith('_reference') else value[key]
        for key in QualityProtocol.__dataclass_fields__ if key not in ('scenarios', 'seeds', 'arms')})


def load_protocol(path=None):
    return parse_protocol(json.loads((Path(path) if path else Path(__file__).with_name('protocol.json')).read_text('utf-8')))
