"""Separate scores, point decisions and event evaluation without point adjustment."""

from dataclasses import dataclass

import numpy as np

from .forecasting_protocols import vector
from ..protocols.schema import ProtocolError, integer


TSB_AD_REVISION = '303dffbfc11612995b17b06633f8c9d0ea0e38e5'
TSB_AD_SUITES = ('TSB-AD-U', 'TSB-AD-M')


def _binary(labels, name):
    labels = vector(labels, name)
    if np.any((labels != 0) & (labels != 1)):
        raise ProtocolError(f'{name} must be a binary mask')
    return labels.astype(int)


def _scores_and_labels(scores, labels, direction):
    scores, labels = vector(scores, 'scores'), _binary(labels, 'labels')
    if scores.shape != labels.shape or direction not in ('higher', 'lower'):
        raise ProtocolError('Detection needs aligned scores and declared score direction')
    return (scores if direction == 'higher' else -scores), labels


def average_precision(scores, labels, *, direction):
    """Point average precision, not trapezoidal PR AUC or range PR AUC."""
    from sklearn.metrics import average_precision_score

    scores, labels = _scores_and_labels(scores, labels, direction)
    if not labels.any():
        return {'status': 'not_available', 'reason': 'no positive labels'}
    return {'status': 'success', 'average_precision': float(average_precision_score(labels, scores))}


def threshold_from_train(scores, *, quantile, direction):
    scores = vector(scores, 'training scores')
    if isinstance(quantile, bool) or not 0 < quantile < 1 or direction not in ('higher', 'lower'):
        raise ProtocolError('A train-only quantile and a valid score direction are required')
    return float(np.quantile(scores if direction == 'higher' else -scores, quantile))


def score_point_mask(scores, labels, *, threshold, direction):
    scores, labels = _scores_and_labels(scores, labels, direction)
    if not np.isfinite(threshold):
        raise ProtocolError('The fixed threshold must be finite')
    found = scores >= threshold
    tp = int((found & (labels == 1)).sum())
    fp = int((found & (labels == 0)).sum())
    fn = int((~found & (labels == 1)).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None}


def score_tsb_ad(scores, labels, *, subset, direction, window, thresholds, reference, reference_revision):
    """Call the pinned upstream range implementation, not a local approximation."""
    if subset not in TSB_AD_SUITES or reference_revision != TSB_AD_REVISION:
        raise ProtocolError('TSB-AD requires its explicit subset and pinned reference revision')
    integer(window, 'VUS window')
    integer(thresholds, 'VUS thresholds', minimum=2)
    scores, labels = _scores_and_labels(scores, labels, direction)
    if len(np.unique(labels)) < 2:
        return {'status': 'not_available', 'reason': 'VUS-PR needs anomaly and normal points'}
    if reference is None:
        return {'status': 'not_available', 'reason': 'pinned TSB-AD evaluator not supplied'}
    vus_pr = float(reference(labels, scores, window, 'opt', thresholds)[-1])
    if not np.isfinite(vus_pr) or not 0 <= vus_pr <= 1:
        raise ProtocolError('Upstream VUS-PR is outside [0, 1] or non-finite')
    return {'status': 'success', 'vus_pr': vus_pr, 'subset': subset, 'reference': reference_revision,
            'window': window, 'thresholds': thresholds}


@dataclass(frozen=True)
class EventWindow:
    start: int
    end: int

    def __post_init__(self):
        integer(self.start, 'event start', minimum=0)
        integer(self.end, 'event end', minimum=0)
        if self.end < self.start:
            raise ProtocolError('Event window end precedes start')


def score_events(windows, alarms, *, length, sample_seconds=1.0):
    """Count first hits and delays in declared non-overlapping inclusive windows."""
    integer(length, 'observation length')
    if not np.isfinite(sample_seconds) or sample_seconds <= 0:
        raise ProtocolError('Observation step must be finite and positive')
    windows = sorted(windows, key=lambda window: window.start)
    indices = tuple(alarms)
    for point in indices:
        integer(point, 'alarm', minimum=0)
    if (len(indices) != len(set(indices)) or any(point >= length for point in indices)
            or any(window.end >= length for window in windows)
            or any(left.end >= right.start for left, right in zip(windows, windows[1:]))):
        raise ProtocolError('Invalid alarm coordinates or overlapping event windows')
    cursor, outside, repeats = 0, 0, 0
    ordered = sorted(indices)
    delays = []
    for window in windows:
        while cursor < len(ordered) and ordered[cursor] < window.start:
            outside += 1
            cursor += 1
        if cursor < len(ordered) and ordered[cursor] <= window.end:
            delays.append((ordered[cursor] - window.start) * sample_seconds)
            cursor += 1
            while cursor < len(ordered) and ordered[cursor] <= window.end:
                repeats += 1
                cursor += 1
    outside += len(ordered) - cursor
    return {'found': len(delays), 'missed': len(windows) - len(delays), 'delay_seconds': tuple(delays),
            'false_alarms': outside, 'repeated_alarms': repeats,
            'false_alarms_per_hour': outside * 3600 / (length * sample_seconds)}
