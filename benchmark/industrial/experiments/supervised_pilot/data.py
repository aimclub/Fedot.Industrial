"""Build and isolate ordered TRAIN panels before any search starts."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ...datasets.local_io import _load_tabular_split, load_local_ts_cases
from ...protocols.assets import AssetIncompatible, LocalAsset, local_root, read_checked_asset
from ...protocols.schema import ProtocolError
from .protocol import PilotProtocol, SplitProfile, TrainSource


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(payload):
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class InnerFold:
    fit: tuple[int, ...]
    score: tuple[int, ...]
    epoch_fit: tuple[int, ...]
    epoch_validation: tuple[int, ...]


@dataclass(frozen=True)
class DataSplits:
    development: tuple[int, ...]
    assessment: tuple[int, ...]
    inner: tuple[InnerFold, ...]
    final_epoch_fit: tuple[int, ...]
    final_epoch_validation: tuple[int, ...]

    @property
    def identity(self):
        return digest(canonical_bytes(asdict(self)))


@dataclass(frozen=True)
class Panel:
    dataset: str
    rows: np.ndarray
    features: np.ndarray
    target: np.ndarray

    def take(self, rows):
        indices = np.searchsorted(self.rows, rows)
        if np.any(indices >= len(self.rows)) or not np.array_equal(self.rows[indices], rows):
            raise ProtocolError('An object key is unavailable in this data partition')
        return Panel(self.dataset, self.rows[indices], self.features[indices], self.target[indices])


@dataclass(frozen=True)
class PreparedDataset:
    source: TrainSource
    panel: Panel
    classes: tuple[str, ...]
    splits: DataSplits


def _snapshot(root, asset):
    root = local_root(root)
    candidate = root / asset.path
    if candidate.resolve() != candidate:
        raise AssetIncompatible('Declared input must not be redirected to a different file')
    return read_checked_asset(root, asset)


def load_train(root, source: TrainSource):
    path, snapshot = _snapshot(root, source.asset)
    if path.suffix == '.tsv':
        values, labels = _load_tabular_split(path, separator='\t', content=snapshot)
        values = values[:, None, :]
    else:
        cases, labels, metadata = load_local_ts_cases(path, content=snapshot)
        if (metadata['target_type'] != 'classification' or not metadata['uniform_channel_layout']
                or metadata['timestamps']):
            raise ProtocolError('TSC pilot requires a uniform, non-timestamped classification panel')
        values = np.asarray(cases, dtype=float)
    if values.shape != source.shape or not np.isfinite(values).all():
        raise ProtocolError(f'{source.name}: invalid panel shape or non-finite values')
    if labels.shape != (values.shape[0],) or any(item is None or item != item for item in labels):
        raise ProtocolError(f'{source.name}: missing or malformed class label')
    labels = labels.astype(str)
    if any(not item.strip() for item in labels):
        raise ProtocolError(f'{source.name}: empty class label')
    return Panel(source.name, np.arange(len(labels)), values, labels)


def _ids(values):
    return tuple(sorted(int(i) for i in values))


def _holdout(profile, rows, target, seed):
    train, score = next(StratifiedShuffleSplit(n_splits=1, test_size=profile.holdout, random_state=seed)
                        .split(rows, target[rows]))
    return np.sort(rows[train]), np.sort(rows[score])


def _partition_pair(left, right, complete):
    if set(left) & set(right) or sorted((*left, *right)) != sorted(complete):
        raise ProtocolError('Split roles overlap, omit or repeat object keys')


def tsc_splits(target, profile: SplitProfile):
    if version('scikit-learn') != profile.sklearn:
        raise ProtocolError(f'TSC split requires scikit-learn=={profile.sklearn}')
    rows = np.arange(len(target))
    dev, assessment = _holdout(profile, rows, target, profile.split_seed)
    _partition_pair(dev, assessment, rows)
    splitter = StratifiedKFold(profile.folds, shuffle=True, random_state=profile.split_seed)
    inner = []
    for fit_index, score_index in splitter.split(dev, target[dev]):
        fit, score = dev[fit_index], dev[score_index]
        epoch_fit, epoch_score = _holdout(profile, fit, target, profile.epoch_seed)
        _partition_pair(fit, score, dev)
        _partition_pair(epoch_fit, epoch_score, fit)
        inner.append(InnerFold(*map(_ids, (fit, score, epoch_fit, epoch_score))))
    final_fit, final_score = _holdout(profile, dev, target, profile.epoch_seed)
    _partition_pair(final_fit, final_score, dev)
    result = DataSplits(_ids(dev), _ids(assessment), tuple(inner), _ids(final_fit), _ids(final_score))
    sets = [result.development, result.assessment, result.final_epoch_fit, result.final_epoch_validation]
    sets.extend(ids for fold in inner for ids in asdict(fold).values())
    classes = set(target)
    if any(set(target[list(rows)]) != classes for rows in sets):
        raise ProtocolError('Every split role must contain all development classes')
    if sorted(row for fold in inner for row in fold.score) != list(result.development):
        raise ProtocolError('Inner score sets must partition the development objects')
    return result


def prepare_tsc(root, source: TrainSource, profile: SplitProfile):
    panel = load_train(root, source)
    splits = tsc_splits(panel.target, profile)
    classes = tuple(sorted(set(panel.take(splits.development).target)))
    if len(classes) != source.classes or len(splits.development) != source.development_size:
        raise ProtocolError(f'{source.name}: unexpected class count or development size')
    if splits.identity != source.split_sha256:
        raise ProtocolError(f'{source.name}: split SHA-256 does not match the protocol')
    return PreparedDataset(source, panel, classes, splits)


def _panel_bytes(panel):
    data = BytesIO()
    np.savez_compressed(data, rows=panel.rows, features=panel.features, target=panel.target)
    return data.getvalue()


def _write_role(directory, panel, description):
    directory.mkdir()
    payload = _panel_bytes(panel)
    (directory / 'panel.npz').write_bytes(payload)
    receipt = {**description, 'dataset': panel.dataset, 'panel_sha256': digest(payload)}
    (directory / 'receipt.json').write_bytes(canonical_bytes(receipt))
    return receipt


def seal_tsc(protocol: PilotProtocol, data_root, output):
    """Verify all sources first; publish the bundle receipt only after all roles are saved."""
    prepared = tuple(prepare_tsc(data_root, source, protocol.profile) for source in protocol.datasets)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for item in prepared:
        dataset_path = root / item.source.name
        dataset_path.mkdir()
        split_metadata = asdict(item.splits)
        assessment = split_metadata.pop('assessment')
        reference = {'train_sha256': item.source.asset.sha256, 'split_sha256': item.splits.identity,
                     'classes': item.classes}
        dev = _write_role(dataset_path / 'development', item.panel.take(item.splits.development),
                          {**reference, 'role': 'development', **split_metadata})
        score = _write_role(dataset_path / 'assessment', item.panel.take(assessment),
                            {**reference, 'role': 'assessment', 'assessment': assessment})
        receipts.append({'dataset': item.source.name, 'development': digest(canonical_bytes(dev)),
                         'assessment': digest(canonical_bytes(score))})
    receipt = {'protocol': protocol.version, 'protocol_sha256': digest(canonical_bytes(protocol.as_payload())),
               'official_test_used': False, 'datasets': receipts}
    (root / 'seal.json').write_bytes(canonical_bytes(receipt))
    return receipt


def read_development(directory, receipt_sha256):
    """Read one authorized development directory, never the bundle or source catalog."""
    _, metadata = _snapshot(directory, LocalAsset('receipt.json', receipt_sha256))
    description = json.loads(metadata)
    if description.get('role') != 'development':
        raise ProtocolError('A search accepts only a development receipt')
    _, payload = _snapshot(directory, LocalAsset('panel.npz', description['panel_sha256']))
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        panel = Panel(description['dataset'], archive['rows'], archive['features'], archive['target'])
    if tuple(panel.rows) != tuple(description['development']):
        raise ProtocolError('Panel object keys do not match the development receipt')
    return panel, description
