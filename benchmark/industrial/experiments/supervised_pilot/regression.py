"""TSER target, split and evaluation contracts independent from class labels."""

from dataclasses import dataclass
from enum import Enum
from functools import partial

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, TimeSeriesSplit
from torch import nn

from ...datasets.local_io import _ordered_axis, _parse_timestamped_dimension, _split_ts_dimension_parts, load_local_ts_cases
from ...protocols.schema import ProtocolError
from .data import DataSplits, InnerFold, Panel, _ids, _snapshot, digest
from .training import NeuralClassifier


class Sampling(str, Enum):
    INDEPENDENT = 'independent'
    GROUPED = 'grouped'
    TEMPORAL = 'temporal'


@dataclass(frozen=True)
class SamplingEvidence:
    kind: Sampling
    source: str
    rows: tuple[int, ...]
    groups: tuple[str, ...] = ()
    times: tuple[int, ...] = ()
    gap: int = 0

    def __post_init__(self):
        if not self.source.strip() or self.rows != tuple(range(len(self.rows))):
            raise ProtocolError('Supply provenance and source-ordered object keys')
        if self.kind is Sampling.GROUPED and (
            len(self.groups) != len(self.rows) or any(not isinstance(group, str) or not group for group in self.groups)
        ):
            raise ProtocolError('Grouped sampling requires a group per object')
        if self.kind is Sampling.TEMPORAL and (
            len(self.times) != len(self.rows) or len(self.times) != len(set(self.times))
            or type(self.gap) is not int or self.gap < 0
        ):
            raise ProtocolError('Temporal sampling requires unique ordered timestamps and a nonnegative gap')
        if not isinstance(self.kind, Sampling):
            raise ProtocolError('Unknown sampling contract')


def _train_snapshot(root, name, asset):
    from pathlib import PurePosixPath

    if PurePosixPath(asset.path).name != f'{name}_TRAIN.ts':
        raise ProtocolError('Declare a separate TSER TRAIN file')
    return _snapshot(root, asset)


def _read_panel(path, content, name):
    cases, target, metadata = load_local_ts_cases(path, content=content)
    if metadata['target_type'] != 'regression' or not metadata['uniform_channel_layout']:
        raise ProtocolError('Expected a uniform time-series panel with a scalar regression target')
    features = np.asarray(cases, dtype=float)
    if target.shape != (len(features),) or not np.isfinite(target).all() or not np.isfinite(features).all():
        raise ProtocolError('TSER expects one finite real target per finite case')
    return Panel(name, np.arange(len(target)), features, target)


def load_regression_train(root, name, asset):
    return _read_panel(*_train_snapshot(root, name, asset), name)


def exact_series_evidence(panel, source):
    """Keep exact input duplicates together independently of target or source row position."""
    features = panel.features.astype('<f8', copy=True)
    features[features == 0] = 0.0
    groups = tuple(digest(row.tobytes(order='C')) for row in features)
    return SamplingEvidence(Sampling.GROUPED, source, tuple(panel.rows.tolist()), groups=groups)


def observed_temporal_panel(root, name, asset):
    """Use absolute windows from TRAIN; overlapping or unaligned inputs require a separate design."""
    path, content = _train_snapshot(root, name, asset)
    panel = _read_panel(path, content, name)
    records = iter(content.decode('utf-8-sig').splitlines())
    if not any(line.strip().lower() == '@data' for line in records):
        raise ProtocolError('Missing TRAIN observations')
    bounds = []
    for line in (row.strip() for row in records):
        if not line or line.startswith('#'):
            continue
        dimensions = _split_ts_dimension_parts(line.rsplit(':', 1)[0], timestamps=True)
        axes = [_ordered_axis(_parse_timestamped_dimension(row, path=path)[0]) for row in dimensions]
        if (not axes or not axes[0] or axes[0][0] not in {'datetime_naive', 'datetime_utc'}
                or any(axis != axes[0] for axis in axes[1:])):
            raise ProtocolError('Expected one absolute aligned time grid for the case')
        _, dates = axes[0]
        if len(dates) != panel.features.shape[2] or any(right <= left for left, right in zip(dates, dates[1:])):
            raise ProtocolError('Expected strictly increasing sample times')
        bounds.append((dates[0], dates[-1]))
    if len(bounds) != len(panel.rows):
        raise ProtocolError('Absolute windows do not match the TRAIN object keys')
    ordered = sorted(bounds)
    if any(previous[1] >= current[0] for previous, current in zip(ordered, ordered[1:])):
        raise ProtocolError('Overlapping TSER windows require a separate verified split rule')
    evidence = SamplingEvidence(Sampling.TEMPORAL, asset.sha256, tuple(panel.rows.tolist()),
                                times=tuple(start for start, _ in bounds))
    return panel, evidence


def regression_splits(panel, evidence: SamplingEvidence):
    if tuple(panel.rows) != evidence.rows:
        raise ProtocolError('Sampling evidence does not match exactly the loaded objects')
    groups = np.array(evidence.groups) if evidence.groups else None

    def holdout(rows, seed):
        if evidence.kind is Sampling.GROUPED:
            fit, score = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                              .split(rows, groups=groups[rows]))
        elif evidence.kind is Sampling.TEMPORAL:
            count = (len(rows) + 4) // 5
            boundary = len(rows) - count
            fit = np.arange(max(0, boundary - evidence.gap))
            score = np.arange(boundary, len(rows))
        else:
            fit, score = next(ShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(rows))
        if not len(fit) or not len(score):
            raise ProtocolError('Insufficient objects for this sampling design')
        left, right = rows[fit], rows[score]
        if evidence.kind is Sampling.TEMPORAL:
            return left, right
        return np.sort(left), np.sort(right)

    order = (panel.rows[np.argsort(evidence.times)] if evidence.kind is Sampling.TEMPORAL else panel.rows)
    dev, score = holdout(order, 42)
    if evidence.kind is Sampling.GROUPED:
        splits = GroupKFold(3).split(dev, groups=groups[dev])
    elif evidence.kind is Sampling.TEMPORAL:
        splits = TimeSeriesSplit(3, gap=evidence.gap).split(dev)
    else:
        splits = KFold(3, shuffle=True, random_state=42).split(dev)
    folds = tuple(InnerFold(_ids(dev[fit]), _ids(dev[val]), *map(_ids, holdout(dev[fit], 2026))) for fit, val in splits)
    return DataSplits(_ids(dev), _ids(score), folds, *map(_ids, holdout(dev, 2026)))


def score_regression(panel, object_keys, prediction):
    predicted = np.asarray(prediction, dtype=float)
    if (not np.array_equal(panel.rows, object_keys) or predicted.shape != panel.target.shape
            or predicted.ndim != 1 or not np.isfinite(predicted).all()):
        raise ProtocolError('Regression predictions must be a finite vector in target object order')
    error = panel.target - predicted
    variance = np.sum((panel.target - panel.target.mean()) ** 2)
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                r2=None if variance == 0 else float(1 - np.sum(error ** 2) / variance))


class NeuralRegressor(NeuralClassifier):
    def __init__(self, family, params, policy=None):
        super().__init__(family, {**params, 'class_count': 1}, policy)

    def _criterion(self):
        return nn.MSELoss()

    def _target(self, target):
        if not np.isfinite(target).all():
            raise ValueError('Regression targets must be finite')
        return target.astype(np.float32)[:, None]

    def predict_proba(self, features):
        raise ValueError('Regression does not produce class probabilities')

    def predict(self, features):
        return self._outputs(features)[:, 0].numpy()


def regression_model_specs():
    from fedot.core.repository.dataset_types import DataTypesEnum as D
    from fedot.core.repository.tasks import TaskTypesEnum as T
    from fedot.extensions import ExternalModelSpec, ModelCapabilities, ModelHyperparamsSchema

    fields = ('batch_size', 'channels', 'device', 'epoch_validation', 'fit_rows', 'learning_rate',
              'length', 'partition_sha256', 'representation_sha256', 'seed')
    return tuple(ExternalModelSpec(
        name, partial(NeuralRegressor, family),
        ModelCapabilities((T.regression,), (data_type,), tags=('neural',), output_data_type=D.tabular),
        ModelHyperparamsSchema(required=fields),
    ) for name, family, data_type in (
        ('cnn2d_regressor', 'cnn2d_classifier', D.image),
        ('tser_inception', 'tsc_inception', D.ts),
        ('tser_resnet', 'tsc_resnet', D.ts),
    ))
