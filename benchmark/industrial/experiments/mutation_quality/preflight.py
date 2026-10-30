"""Read TRAIN sources and construct split plans without importing an optimizer."""

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .config import load_protocol


def file_digest(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def supervised_splits(target, task, protocol):
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
    indices = np.arange(len(target))
    search, outer = train_test_split(indices, test_size=protocol.outer_fraction, random_state=protocol.split_seed,
                                     stratify=target if task == 'classification' else None)
    search, outer = np.sort(search), np.sort(outer)
    if task == 'classification':
        _, counts = np.unique(target[search], return_counts=True)
        if len(counts) < 2 or counts.min() < protocol.inner_folds:
            raise ValueError('Insufficient search samples per class for stratified inner folds')
        splitter = StratifiedKFold(protocol.inner_folds, shuffle=True, random_state=protocol.split_seed)
    else:
        splitter = KFold(protocol.inner_folds, shuffle=True, random_state=protocol.split_seed)
    folds = [dict(train_indices=search[left].tolist(), validation_indices=search[right].tolist())
             for left, right in splitter.split(search, target[search])]
    return dict(search_indices=search.tolist(), outer_indices=outer.tolist(), folds=folds,
                normalization='fit channel mean/std on each inner training partition only')


def temporal_splits(length, protocol):
    horizon = protocol.forecast_horizon
    search_end = length - horizon
    boundaries = range(search_end - protocol.inner_folds * horizon, search_end, horizon)
    if search_end - protocol.inner_folds * horizon <= 24:
        raise ValueError('Temporal TRAIN history is too short for the unchanged window')
    return dict(search_end=search_end, outer_start=search_end, outer_end=length, horizon=horizon,
                folds=[dict(train_end=end, validation_start=end, validation_end=end + horizon) for end in boundaries])


def first_eligible_series(path, minimum_length):
    # M4 wide CSV: first field is the series ID, padding is allowed only at the right.
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        rows = csv.reader(stream)
        next(rows)
        for row in rows:
            if not row:
                continue
            values = list(row[1:])
            while values and values[-1].strip().lower() in ('', 'nan', 'na'):
                values.pop()
            if any(value.strip().lower() in ('', 'nan', 'na') for value in values):
                continue
            array = np.asarray(values, dtype=float)
            if len(array) >= minimum_length and np.isfinite(array).all():
                return row[0], array
    raise ValueError('No eligible finite TRAIN series in source order')


def read_source(scenario, root, protocol):
    path = (root / scenario.path).resolve() if scenario.path else None
    if path is not None and not path.is_relative_to(root.resolve()):
        raise ValueError('Resolved source escapes repository root')
    source = {}
    if path is not None:
        source = dict(path=scenario.path, sha256=file_digest(path))
    if scenario.source == 'pending_external':
        raise FileNotFoundError('External TRAIN source and format are not configured')
    if scenario.source == 'tsv':
        values = np.loadtxt(path, delimiter='\t', ndmin=2)
        if values.shape[1] < 2:
            raise ValueError('TSV must contain a target and features')
        x, y = values[:, None, 1:], values[:, 0]
    elif scenario.source == 'ts':
        from fedot_ind.tools.ts_parser import parse_ts_file
        frame, y = parse_ts_file(str(path), encoding='utf-8')
        x = np.stack([np.stack([np.asarray(channel, dtype=float) for channel in row])
                      for row in frame.itertuples(index=False, name=None)])
    elif scenario.source == 'sktime_tecator':
        from sktime.datasets import load_tecator
        import sktime
        x, y = load_tecator(split='train', return_type='numpy3d')
        source = dict(provider='sktime.datasets.load_tecator', split='train', sktime_version=sktime.__version__)
    elif scenario.source == 'wide_csv':
        series_id, x = first_eligible_series(path, protocol.minimum_series_length)
        source.update(series_id=series_id, selection='first finite eligible TRAIN row in file order')
        return x, None, source
    elif scenario.source == 'airline_snapshot':
        manifest_path = path.with_name('manifest.json')
        entries = json.loads(manifest_path.read_text('utf-8'))
        entry = next(item for item in entries if item['file'] == path.name)
        if entry['sha256'] != source['sha256'] or entry['official_test_used'] is not False:
            raise ValueError('Historical TRAIN snapshot integrity or role mismatch')
        with np.load(path, allow_pickle=False) as saved:
            x = np.concatenate((saved['refit_features'].reshape(-1), saved['control_target'].reshape(-1)))
        source.update(manifest_sha256=file_digest(manifest_path), role='historical official TRAIN reconstructed')
        return x, None, source
    else:
        raise ValueError('Unsupported source')
    return np.asarray(x, dtype=float), np.asarray(y), source


def inspect_scenario(scenario, root, protocol):
    row = dict(name=scenario.name, task=scenario.task, group=scenario.group, official_test_used=False)
    try:
        x, y, source = read_source(scenario, root, protocol)
        if scenario.path and file_digest(root / scenario.path) != source['sha256']:
            raise ValueError('TRAIN source changed while being read')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite features require an explicit preprocessing decision')
        digest = sha256()
        digest.update(str((x.shape, x.dtype.str)).encode())
        digest.update(x.tobytes())
        row.update(source=source, shape=list(x.shape), dtype=str(x.dtype), features_sha256=digest.hexdigest())
        if scenario.task == 'ts_forecasting':
            if x.ndim != 1 or len(x) < protocol.minimum_series_length:
                raise ValueError('Invalid forecasting series shape or length')
            split = temporal_splits(len(x), protocol)
        else:
            if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
                raise ValueError('Expected samples/channels/time with one target per sample')
            if scenario.task == 'regression':
                y = y.astype(float)
                if not np.isfinite(y).all():
                    raise ValueError('Nonfinite regression target')
            else:
                labels, counts = np.unique(y, return_counts=True)
                row['classes'] = dict(zip(map(str, labels), map(int, counts)))
            row['target_sha256'] = sha256(json.dumps(y.tolist(), ensure_ascii=True,
                                          allow_nan=False).encode()).hexdigest()
            split = supervised_splits(y, scenario.task, protocol)
        row.update(status='ready_for_split_review', split=split)
    except (FileNotFoundError, ImportError) as error:
        row.update(status='missing_input', error_type=type(error).__name__, reason=str(error))
    except ValueError as error:
        row.update(status='invalid_input', error_type=type(error).__name__, reason=str(error))
    return row


def preview(protocol):
    return dict(training_enabled=False, jobs=len(protocol.jobs()), scenarios=len(protocol.scenarios),
                search_budget_hours=len(protocol.jobs()) * protocol.budget_seconds / 3600,
                arms=[arm.value for arm in protocol.arms], seeds=list(protocol.seeds),
                forecast_horizon=protocol.forecast_horizon,
                launch_blockers=['No training command is implemented in this preparation package',
                                 'A must exclude rewrite actions; verify runtime dispatch before training',
                                 'Review and freeze source hashes, selected series, splits and all runtime dependencies'])


def preflight(root, protocol=None):
    protocol = protocol or load_protocol()
    result = preview(protocol)
    result['protocol'] = asdict(protocol)
    result['scenarios'] = [inspect_scenario(row, Path(root), protocol) for row in protocol.scenarios]
    result['references'] = {}
    for name, path in [('training', protocol.training_reference), ('policy', protocol.policy_reference)]:
        reference = Path(root) / path
        result['references'][name] = {'path': path, 'sha256': file_digest(reference) if reference.is_file() else None}
        if not reference.is_file():
            result['launch_blockers'].append(f'Missing required reference: {path}')
    result['jobs'] = protocol.jobs()
    result['matrix_ready'] = all(row['status'] == 'ready_for_split_review' for row in result['scenarios'])
    return result
