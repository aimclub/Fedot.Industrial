"""Verify DATA-06 on pinned TSB-AD releases and a local NAB stream."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from math import isclose
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

from benchmark.industrial.evaluation.detection_protocols import (
    EventWindow, TSB_AD_REVISION, TSB_AD_SUITES, score_events, score_tsb_ad, threshold_from_train,
)
from benchmark.industrial.protocols.assets import LocalAsset, checked_asset, file_digest
from benchmark.industrial.protocols.json_io import read_protocol_json


def _asset(root, entry):
    return checked_asset(root, LocalAsset(entry['path'], entry['sha256']))


def _file_list(root, entry):
    names = pd.read_csv(_asset(root, entry))['file_name']
    if len(names) != entry['count'] or names.isna().any() or not names.is_unique:
        raise ValueError('Published TSB-AD list has unexpected membership')
    return frozenset(names)


def _check_split_lists(root, profile):
    names, evaluation, tuning = (_file_list(root, profile[part]) for part in ('all', 'evaluation', 'tuning'))
    full = _file_list(root, profile['evaluation_full']) if 'evaluation_full' in profile else evaluation
    if not evaluation <= full or full & tuning or full | tuning != names:
        raise ValueError('TSB-AD release differs from the published tuning/evaluation lists')
    return names, evaluation


def _load_tsb_example(root, profile):
    example = profile['example']
    names, evaluation = _check_split_lists(root, profile)
    if PurePosixPath(example['path']).name not in evaluation:
        raise ValueError('Diagnostic example is not in the official evaluation list')
    with ZipFile(_asset(root, profile['archive'])) as archive:
        members = [PurePosixPath(path).name for path in archive.namelist()
                   if path.endswith('.csv') and not path.startswith('__MACOSX/')]
        if len(members) != len(names) or set(members) != names:
            raise ValueError('TSB-AD archive differs from its published file list')
        sample = archive.read(example['path'])
    if sha256(sample).hexdigest() != example['sha256']:
        raise ValueError('TSB-AD example SHA-256 differs from the reference')
    return pd.read_csv(BytesIO(sample))


def _causal_delta(values):
    return np.abs(np.diff(values, axis=0, prepend=values[:1]))


def _tsb_example_scores(frame, example):
    labels = frame.pop('Label').to_numpy()
    values = frame.to_numpy(dtype=float)
    count = example['train_count']
    if (values.shape != (example['rows'], example['channels']) or not np.isfinite(values).all()
            or not 1 < count < len(labels) or np.any(labels[:count]) or not np.isin(labels, (0, 1)).all()):
        raise ValueError('TSB-AD example does not satisfy the declared cardinality and label contract')
    # Scale each channel using only differences inside the training prefix.
    deltas = _causal_delta(values)
    scale = deltas[1:count].mean(axis=0)
    scores = (deltas / np.where(scale > 0, scale, 1)).mean(axis=1)
    return scores[count:], labels[count:].astype(int)


def verify_tsb_ad(root, config):
    from TSB_AD.evaluation import basic_metrics

    if (config['revision'] != TSB_AD_REVISION or set(config['subsets']) != set(TSB_AD_SUITES)
            or file_digest(Path(basic_metrics.__file__)) != config['metric_sha256']):
        raise ValueError('Installed TSB-AD evaluator does not match the pinned reference')
    results = {}
    for subset, profile in config['subsets'].items():
        example = profile['example']
        frame = _load_tsb_example(root, profile)
        scores, labels = _tsb_example_scores(frame, example)
        result = score_tsb_ad(scores, labels, subset=subset, direction='higher',
                              window=config['window'], thresholds=config['thresholds'],
                              reference=basic_metrics.generate_curve, reference_revision=config['revision'])
        _, _, _, _, _, reference = basic_metrics.basic_metricor().RangeAUC_volume_opt_mem(
            labels, scores, config['window'], thre=config['thresholds'])
        np.testing.assert_allclose([result['vus_pr']] * 2, [reference, example['expected_vus_pr']], rtol=0, atol=1e-12)
        results[subset] = {**result, 'archive': profile['archive'], 'example': example['path'],
                           'all_series': profile['all']['count'], 'evaluation_series': profile['evaluation']['count'],
                           'tuning_series': profile['tuning']['count'], 'channels': example['channels'],
                           'test_points': len(labels), 'anomalous_points': int(labels.sum()),
                           'reference_opt_mem': float(reference)}
    return results


def _elapsed_slots(timestamps, origin, step):
    offsets = pd.DatetimeIndex(timestamps) - origin
    if offsets.hasnans or not (offsets % step == pd.Timedelta(0)).all():
        raise ValueError('NAB timestamps do not align with the declared time grid')
    return tuple(int(index) for index in (offsets // step))


def _timestamp_reference(windows, alarms, start, last, step):
    """Independent timestamp membership check; it never uses row positions."""
    inside = [tuple(time for time in alarms if left <= time <= right) for left, right in windows]
    first = tuple(points[0] if points else None for points in inside)
    delay = tuple((time - left).total_seconds() for (left, _), time in zip(windows, first, strict=True)
                  if time is not None)
    false = sum(not any(left <= time <= right for left, right in windows) for time in alarms)
    repeats = sum(max(0, len(points) - 1) for points in inside)
    exposure_hours = (last - start + step).total_seconds() / 3600
    return first, delay, false, repeats, exposure_hours


def verify_nab(root, config):
    assets = {role: _asset(root, entry) for role, entry in config['assets'].items()}
    stream = pd.read_csv(assets['stream'], parse_dates=['timestamp'])
    times = pd.DatetimeIndex(stream['timestamp'])
    values = stream['value'].to_numpy(dtype=float)
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing or not np.isfinite(values).all():
        raise ValueError('NAB requires finite observations and ordered unique timestamps')
    windows = tuple(tuple(pd.Timestamp(time) for time in pair)
                    for pair in read_protocol_json(assets['windows'])[config['key']])
    labels = tuple(pd.Timestamp(time) for time in read_protocol_json(assets['labels'])[config['key']])
    count = int(len(times) * config['train_fraction'])
    step = pd.Timedelta(seconds=config['sample_seconds'])
    if step <= pd.Timedelta(0) or not 1 < count < len(times):
        raise ValueError('NAB requires positive cadence and non-empty training and evaluation periods')
    start, last = times[count], times[-1]
    if any(not start <= left <= label <= right <= last for (left, right), label in zip(windows, labels, strict=True)):
        raise ValueError('NAB windows and labels must match and lie entirely in the scoring period')
    scores = _causal_delta(values)
    threshold = threshold_from_train(scores[:count], quantile=config['alarm_quantile'], direction='higher')
    alert = scores[count:] >= threshold
    alarm_times = times[count:][alert]
    slots = _elapsed_slots(times[count:], start, step)
    alarm_slots = [slot for slot, raised in zip(slots, alert, strict=True) if raised]
    event_windows = [EventWindow(*_elapsed_slots(pair, start, step)) for pair in windows]
    length = slots[-1] + 1
    result = score_events(event_windows, alarm_slots, length=length, sample_seconds=config['sample_seconds'])
    first, delay, false, repeats, exposure_hours = _timestamp_reference(windows, alarm_times, start, last, step)
    if (result['found'] != len(delay) or result['missed'] != len(windows) - len(delay)
            or result['delay_seconds'] != delay or result['false_alarms'] != false
            or result['repeated_alarms'] != repeats or not isclose(result['false_alarms_per_hour'],
                                                                   false / exposure_hours, rel_tol=0, abs_tol=1e-12)
            or len(alarm_times) != result['found'] + false + repeats):
        raise ValueError('NAB event metrics differ from direct timestamp evaluation')
    return {'stream': config['key'], 'revision': config['revision'], 'train_points': count,
            'test_points': len(times) - count, 'unobserved_slots': length - (len(times) - count),
            'threshold': threshold, 'alarm_count': len(alarm_times), 'exposure_hours': exposure_hours,
            'first_alarms': [time.isoformat() if time is not None else None for time in first], **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--suite', choices=('tsb_ad', 'nab', 'all'), default='all')
    args = parser.parse_args(argv)
    config = read_protocol_json(Path(__file__).with_name('detection_metric_acceptance_sources.json'))
    if config['version'] != 'industrial_detection_acceptance@1':
        raise ValueError('Unrecognized identification for DATA-06')
    runners = {'tsb_ad': verify_tsb_ad, 'nab': verify_nab}
    suites = runners if args.suite == 'all' else (args.suite,)
    print(json.dumps({name: runners[name](args.data_root, config[name]) for name in suites},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
