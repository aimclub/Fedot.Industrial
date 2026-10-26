"""Pure validation and deterministic experiment-matrix construction."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re


TASKS = ('classification', 'regression', 'ts_forecasting')


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True)
class FrozenProtocol:
    json_text: str
    digest: str

    def to_record(self):
        return json.loads(self.json_text)


def parse_protocol(raw):
    integer_fields = ('population_size', 'generations', 'search_seconds', 'run_seconds', 'max_candidates',
                      'max_depth', 'max_nodes', 'max_arity', 'mutation_attempts')
    required = set(integer_fields) | {'schema_version', 'seeds', 'supervised_root', 'classification_root',
                                      'forecast_train', 'forecast_test', 'scenarios', 'operations',
                                      'initial_chains', 'operation_parameters', 'acceptance'}
    if not isinstance(raw, dict) or set(raw) != required or raw['schema_version'] != 'industrial_population_study@1':
        raise ValueError('Population study fields/schema must match the contract')
    for name in integer_fields:
        if not isinstance(raw[name], int) or raw[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if raw['generations'] < 4 or raw['population_size'] < 4 or raw['run_seconds'] <= raw['search_seconds']:
        raise ValueError('Require a full population, >=3 evolved generations and a larger overall deadline')
    seeds = raw['seeds']
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('Require at least three unique seeds')
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('seeds must be nonnegative integers')
    for name in ('supervised_root', 'classification_root', 'forecast_train', 'forecast_test'):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f'{name} must be a non-empty path')
    seen, observed = set(), set()
    if not isinstance(raw['scenarios'], list) or not raw['scenarios']:
        raise ValueError('scenarios must be a non-empty list')
    for scenario in raw['scenarios']:
        if (not isinstance(scenario, dict) or set(scenario) != {'name', 'task'} or scenario['task'] not in TASKS
                or not isinstance(scenario['name'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', scenario['name'])):
            raise ValueError('Invalid scenario')
        key = scenario['task'], scenario['name']
        if key in seen:
            raise ValueError('Duplicate scenario')
        seen.add(key)
        observed.add(scenario['task'])
    if observed != set(TASKS):
        raise ValueError('All three tasks must be present')
    if any(not isinstance(raw[name], dict) or set(raw[name]) != set(TASKS)
           for name in ('operations', 'initial_chains')):
        raise ValueError('Operation and chain maps must cover exactly the three tasks')
    for task in TASKS:
        operations = raw['operations'][task]
        if (not isinstance(operations, list) or not operations
                or any(not isinstance(name, str) or not name for name in operations)
                or len(set(operations)) != len(operations)):
            raise ValueError('Operation lists must be non-empty and unique')
        if any('okhs' in name.lower() for name in operations):
            raise ValueError('OKHS-FDMD is excluded from this study')
        chains = raw['initial_chains'][task]
        if not chains or any(not chain or not set(chain) <= set(operations) for chain in chains):
            raise ValueError('Initial chains must belong to the allowlist')
    criteria = raw['acceptance']
    for name in ('quality_tolerance', 'worst_seed_tolerance'):
        if set(criteria[name]) != set(TASKS):
            raise ValueError('Quality criteria must cover all tasks')
    for value in [*criteria['quality_tolerance'].values(), *criteria['worst_seed_tolerance'].values(),
                  *(value for name, value in criteria.items()
                    if name not in ('quality_tolerance', 'worst_seed_tolerance'))]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Acceptance limits must be finite nonnegative numbers')
    encoded = canonical(raw)
    return FrozenProtocol(encoded, sha256(encoded.encode()).hexdigest())


def load_protocol(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name('defaults.json')
    return parse_protocol(json.loads(path.read_text(encoding='utf-8')))


def experiment_matrix(protocol):
    rows = []
    for scenario in protocol.to_record()['scenarios']:
        for seed in protocol.to_record()['seeds']:
            order = ('legacy', 'planned') if len(rows) // 2 % 2 == 0 else ('planned', 'legacy')
            for mode in order:
                rows.append({'ordinal': len(rows), 'scenario': scenario, 'seed': seed, 'mode': mode,
                             'protocol_sha256': protocol.digest})
    return tuple(rows)
