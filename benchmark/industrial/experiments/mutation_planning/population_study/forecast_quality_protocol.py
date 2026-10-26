"""Pure protocol, temporal cuts and selection for exploratory forecast diagnosis."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .protocol import canonical


@dataclass(frozen=True)
class QualityCandidate:
    name: str
    kind: str
    chain: tuple[str, ...] = ()
    parameters_json: str = '{}'
    period: int = 1
    reference_solver: bool = False
    denoised_state: bool = False


def parse_config(raw):
    fields = {'schema_version', 'train', 'test', 'datasets', 'horizon', 'folds', 'seeds',
              'test_comparators', 'candidates'}
    if (type(raw) is not dict or set(raw) != fields
            or raw['schema_version'] != 'industrial_forecast_quality_diagnostic@1'):
        raise ValueError('Invalid forecast quality diagnostic schema')
    for key in ('train', 'test'):
        if type(raw[key]) is not str or not raw[key].strip():
            raise ValueError('Input paths must be nonempty strings')
    for key in ('horizon', 'folds'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError('Horizon and fold count must be positive integers')
    for key in ('seeds', 'datasets'):
        if type(raw[key]) is not list or not raw[key]:
            raise ValueError('Datasets and seeds must be nonempty lists')
    if any(type(seed) is not int or seed < 0 for seed in raw['seeds']):
        raise ValueError('Seeds must be nonnegative integers')
    if any(type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in raw['datasets']):
        raise ValueError('Invalid dataset identifiers')
    if any(len(set(raw[key])) != len(raw[key]) for key in ('seeds', 'datasets')):
        raise ValueError('Datasets and seeds must be unique')
    candidates = []
    allowed = {'name', 'kind', 'chain', 'parameters', 'period', 'reference_solver', 'denoised_state'}
    if type(raw['candidates']) is not list or not raw['candidates']:
        raise ValueError('Require diagnostic candidates')
    for item in raw['candidates']:
        if (type(item) is not dict or not set(item) <= allowed or not {'name', 'kind'} <= set(item)
                or type(item['name']) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+', item['name'])
                or item['kind'] not in ('industrial', 'naive', 'drift')):
            raise ValueError('Invalid diagnostic candidate')
        chain = item.get('chain', [])
        if (type(chain) is not list or any(type(name) is not str for name in chain)
                or (item['kind'] == 'industrial' and chain not in (
                    ['mssa_forecaster'], ['ssa_forecaster'], ['smoothing', 'mssa_forecaster']))
                or (item['kind'] != 'industrial' and chain)):
            raise ValueError('Unsupported chain; this diagnostic excludes other forecasting families')
        if type(item.get('parameters', {})) is not dict:
            raise ValueError('Parameters must be an object')
        if item['kind'] != 'industrial' and item.get('parameters'):
            raise ValueError('Baseline candidates do not consume model parameters')
        if item['kind'] != 'naive' and 'period' in item:
            raise ValueError('Only naive candidates consume a seasonal period')
        if type(item.get('period', 1)) is not int or item.get('period', 1) < 1:
            raise ValueError('Period must be a positive integer')
        for flag in ('reference_solver', 'denoised_state'):
            if type(item.get(flag, False)) is not bool:
                raise ValueError('Diagnostic switches must be booleans')
        if any(item.get(flag, False) for flag in ('reference_solver', 'denoised_state')):
            if item['kind'] != 'industrial' or item.get('parameters', {}).get('head_policy') != 'linear':
                raise ValueError('Reference/state probes require an explicit linear industrial head')
        candidates.append(QualityCandidate(item['name'], item['kind'], tuple(chain),
                                           canonical(item.get('parameters', {})), item.get('period', 1),
                                           item.get('reference_solver', False), item.get('denoised_state', False)))
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate candidate names')
    if (type(raw['test_comparators']) is not list
            or any(type(name) is not str for name in raw['test_comparators'])
            or not set(raw['test_comparators']) <= set(names)
            or len(set(raw['test_comparators'])) != len(raw['test_comparators'])):
        raise ValueError('Unknown or duplicate fixed test comparators')
    canonical(raw)
    return tuple(candidates)


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name('forecast_quality_defaults.json')
    raw = json.loads(path.read_text(encoding='utf-8'))
    return raw, parse_config(raw)


def temporal_cuts(length, horizon, folds):
    cuts = tuple(length - horizon * offset for offset in range(folds, 0, -1))
    if not cuts or cuts[0] < max(4, horizon + 1):
        raise ValueError('Not enough history for the diagnostic temporal folds')
    return cuts


def select_candidates(records, config, candidates):
    """Require complete finite validation evidence; never consume test records."""
    if any(record.get('stage') != 'validation' for record in records):
        raise ValueError('Selection can only consume validation records')
    summaries, selections = [], {}
    expected = {(fold, seed) for fold in range(config['folds']) for seed in config['seeds']}
    for dataset in config['datasets']:
        eligible = []
        for candidate in candidates:
            rows = [row for row in records if row['dataset'] == dataset and row['candidate'] == candidate.name]
            complete = (len(rows) == len(expected) and {(row['fold'], row['seed']) for row in rows} == expected
                        and all(row['status'] == 'succeeded' and np.isfinite(row['mase']) for row in rows))
            score = float(np.mean([row['mase'] for row in rows])) if complete else None
            summaries.append({'dataset': dataset, 'candidate': candidate.name,
                              'complete': complete, 'validation_mase': score,
                              'production_candidate': candidate.kind == 'industrial'
                              and not candidate.reference_solver and not candidate.denoised_state})
            if complete:
                eligible.append((score, candidate.name, candidate.kind))
        if not eligible:
            raise ValueError(f'No complete validation candidate for {dataset}')
        production_names = {candidate.name for candidate in candidates if candidate.kind == 'industrial'
                            and not candidate.reference_solver and not candidate.denoised_state}
        industrial = [item for item in eligible if item[1] in production_names]
        selections[dataset] = {'best': min(eligible)[1],
                               'best_industrial': min(industrial)[1] if industrial else None}
    return summaries, selections
