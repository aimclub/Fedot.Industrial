"""Group-aware admission for published and chronological classification splits."""

from dataclasses import dataclass
from heapq import heappop, heappush
from io import BytesIO

import numpy as np

from ..protocols.schema import ProtocolError, coordinate, coordinate_kind, fields, string


_WINDOW_COORDINATES = ('window_start', 'window_end', 'decision_time')


@dataclass(frozen=True)
class DriftCase:
    case_id: str
    group_id: str
    segment: str
    target: str
    window_start: int | str
    window_end: int | str
    decision_time: int | str
    event_time: int | str | None = None

    def __post_init__(self):
        coordinates = {name: coordinate(getattr(self, name), name) for name in _WINDOW_COORDINATES}
        if self.event_time is not None:
            coordinates['event_time'] = coordinate(self.event_time, 'event_time')
        domain = {coordinate_kind(value) for value in coordinates.values()}
        start, end, decision = (coordinates[name] for name in _WINDOW_COORDINATES)
        if len(domain) != 1 or not start <= end <= decision:
            raise ProtocolError('Drift window crosses the decision or changes coordinate domain')
        for name, value in coordinates.items():
            object.__setattr__(self, name, value)

    @classmethod
    def parse(cls, row):
        names = ('case_id', 'group_id', 'segment', 'target')
        row = fields(row, (*names, *_WINDOW_COORDINATES), 'drift case', ('event_time',))
        return cls(*(string(row[name], name) for name in names),
                   *(row[name] for name in _WINDOW_COORDINATES), row.get('event_time'))


@dataclass(frozen=True)
class DriftSplit:
    mode: str
    train: tuple[DriftCase, ...]
    test: tuple[DriftCase, ...]


def _split(cases, test_ids, mode, early_warning):
    cases = tuple(cases)
    if (not cases or len({case.case_id for case in cases}) != len(cases)
            or len({coordinate_kind(case.decision_time) for case in cases}) != 1):
        raise ProtocolError('Drift cases require unique IDs and one time coordinate domain')
    if early_warning and any(case.event_time is None or case.decision_time >= case.event_time for case in cases):
        raise ProtocolError('Early-warning decisions must precede the events they anticipate')
    train = tuple(case for case in cases if case.case_id not in test_ids)
    test = tuple(case for case in cases if case.case_id in test_ids)
    if not train or not test or {case.case_id for case in test} != test_ids:
        raise ProtocolError('Drift split needs known non-empty train and test cases')
    if {case.group_id for case in train} & {case.group_id for case in test}:
        raise ProtocolError('A patient, trial or incident appears in both partitions')
    return DriftSplit(mode, train, test)


def published_drift_split(cases, partition_by_id, *, early_warning=False):
    """Preserve the source split, including disease-specific held-out weeks."""
    cases = tuple(cases)
    if (set(partition_by_id) != {case.case_id for case in cases}
            or set(partition_by_id.values()) != {'train', 'test'}):
        raise ProtocolError('Published splits must explicitly classify every case as train or test')
    test_ids = {case_id for case_id, part in partition_by_id.items() if part == 'test'}
    return _split(cases, test_ids, 'published', early_warning)


def chronological_drift_split(cases, cutoff, *, early_warning=False):
    cutoff = coordinate(cutoff, 'chronological cutoff')
    cases = tuple(cases)
    if any(coordinate_kind(case.decision_time) != coordinate_kind(cutoff) for case in cases):
        raise ProtocolError('Decision and cutoff use different coordinate types')
    test_ids = {case.case_id for case in cases if case.decision_time >= cutoff}
    split = _split(cases, test_ids, 'chronological', early_warning)
    if any(case.window_start < cutoff for case in split.test):
        raise ProtocolError('A held-out input window crosses the chronological cutoff')
    return split


def overlapping_segments(split: DriftSplit):
    """Count cross-partition window pairs on the same segment, without returning source IDs."""
    active = {}
    total = 0
    tagged = [(case.segment, case.window_start, case.window_end, part)
              for part, cases in enumerate((split.train, split.test)) for case in cases]
    for segment, start, end, part in sorted(tagged):
        heaps = active.setdefault(segment, ([], []))
        other = heaps[1 - part]
        while other and other[0] < start:
            heappop(other)
        total += len(other)
        heappush(heaps[part], end)
    return total


def binary_drift_report(split, scores, *, positive, negative):
    """Publish only aggregate ROC AUC and group counts, not source-level records."""
    from sklearn.metrics import roc_auc_score

    if positive == negative or {case.target for case in (*split.train, *split.test)} - {positive, negative}:
        raise ProtocolError('Select an explicit binary comparison before evaluating drift')
    if set(scores) != {case.case_id for case in split.test}:
        raise ProtocolError('Scores must cover exactly the held-out cases')
    labels = [case.target == positive for case in split.test]
    predictions = np.asarray([scores[case.case_id] for case in split.test], dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ProtocolError('Drift scores must be finite scalars')
    auc = float(roc_auc_score(labels, predictions)) if len(set(labels)) == 2 else None
    return {'mode': split.mode, 'train_count': len(split.train), 'test_count': len(split.test),
            'test_groups': len({case.group_id for case in split.test}), 'positive_count': sum(labels),
            'roc_auc': auc, 'status': 'success' if auc is not None else 'not_available',
            'reason': None if auc is not None else 'held-out set lacks one class'}


def load_hydraulic_cycles(release, root, *, sensors, target_column):
    """Read UCI sensor tables by cycle, without changing sampling rates or inventing trial IDs."""
    from ..protocols.schema import integer, unique_strings

    unique_strings(sensors, 'hydraulic sensor roles')
    integer(target_column, 'hydraulic target column', minimum=0)
    if not sensors or 'profile' in sensors:
        raise ProtocolError('Declare at least one sensor role distinct from profile')

    def read_table(role):
        return np.loadtxt(BytesIO(release.read_bytes(root, role)), ndmin=2)

    profile = read_table('profile')
    channels = {sensor: read_table(sensor) for sensor in sensors}
    if target_column >= profile.shape[1] or any(len(values) != len(profile) for values in channels.values()):
        raise ProtocolError('Hydraulic sensor rows must align with the cycle level profile')
    return channels, profile[:, target_column]
