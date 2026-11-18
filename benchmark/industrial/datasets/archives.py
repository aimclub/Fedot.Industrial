"""Locked UCR, UEA and original TSER releases and offline reference baselines."""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .local_io import load_local_supervised_split, load_local_ts_cases
from ..protocols.assets import AssetCorrupt, AssetIncompatible, AssetUnavailable, LocalAsset, checked_asset
from ..protocols.schema import ProtocolError


@dataclass(frozen=True)
class ArchiveProfile:
    version: str
    task: str
    source: str
    tasks: tuple[str, ...]
    smoke: tuple[str, ...]

    def selection(self, scope: str):
        if scope not in ('full', 'smoke'):
            raise ProtocolError('Archive scope must be full or smoke')
        return self.tasks if scope == 'full' else self.smoke


@lru_cache(maxsize=None)
def archive_profile(version: str) -> ArchiveProfile:
    path = Path(__file__).with_name('archive_profiles.json')
    entries = json.loads(path.read_text(encoding='utf-8'))
    if version not in entries:
        raise ProtocolError(f'Unknown archive release: {version!r}')
    entry = entries[version]
    return ArchiveProfile(version, entry['task'], entry['source'], tuple(entry['tasks']), tuple(entry['smoke']))


@dataclass(frozen=True)
class ArchiveSplit:
    profile: ArchiveProfile
    dataset: str
    train: tuple[tuple[np.ndarray, ...], ...]
    train_target: np.ndarray
    test: tuple[tuple[np.ndarray, ...], ...]
    test_target: np.ndarray
    split_id: str


# Published sizes of the three smoke tasks and the separate prepared PEMS-SF release.
_REFERENCE_COUNTS = {
    ('ucr_2018', 'ECG200'): (100, 100),
    ('uea_2018', 'BasicMotions'): (40, 40),
    ('uea_2018', 'PEMS-SF'): (267, 173),
    # The published Zenodo files (3902637) contain 95 train cases; the paper lists 96.
    ('tser_original_19', 'AppliancesEnergy'): (95, 42),
}


def _read_pair(train, test, task):
    if train.suffix.lower() != test.suffix.lower():
        raise ProtocolError('TRAIN and TEST must use the same file format')
    suffix = train.suffix.lower()
    if suffix == '.ts':
        train_cases, train_target, train_metadata = load_local_ts_cases(train)
        test_cases, test_target, test_metadata = load_local_ts_cases(test)
        target_type = 'regression' if task == 'ts_regression' else 'classification'
        if any(metadata['target_type'] != target_type for metadata in (train_metadata, test_metadata)):
            raise ProtocolError('Declared target kind does not match the archive task')
        if task != 'ts_regression' and (train_metadata['timestamps'] or test_metadata['timestamps']):
            raise ProtocolError('Timestamped Euclidean records require an explicit alignment policy')
        return train_cases, train_target, test_cases, test_target
    if suffix not in ('.tsv', '.csv'):
        raise ProtocolError(f'Unsupported archive pair format: {suffix}')
    local = load_local_supervised_split('locked-archive', train_path=train, test_path=test)
    return (tuple((row,) for row in local.train_features), local.train_target,
            tuple((row,) for row in local.test_features), local.test_target)


def load_archive_split(profile, dataset, root, train: LocalAsset, test: LocalAsset, *, verify_counts=True):
    """Read the declared TRAIN/TEST pair, preserving channel and record boundaries."""
    if dataset not in profile.tasks:
        raise ProtocolError(f'{dataset!r} is not in {profile.version}')
    if train.sha256 == test.sha256:
        raise ProtocolError('TRAIN and TEST must identify distinct contents')
    train_path, test_path = checked_asset(root, train), checked_asset(root, test)
    if train_path.stem != f'{dataset}_TRAIN' or test_path.stem != f'{dataset}_TEST':
        raise ProtocolError('TRAIN and TEST paths do not match the archive dataset and roles')
    train_cases, train_y, test_cases, test_y = _read_pair(train_path, test_path, profile.task)
    counts = _REFERENCE_COUNTS.get((profile.version, dataset))
    if not train_cases or not test_cases or (verify_counts and counts and counts != (len(train_cases), len(test_cases))):
        raise ProtocolError('TRAIN/TEST sizes do not match the declared archive')
    channel_counts = {len(case) for case in (*train_cases, *test_cases)}
    if len(channel_counts) != 1 or 0 in channel_counts:
        raise ProtocolError('Channel count differs across records or partitions')
    if profile.version == 'ucr_2018' and channel_counts != {1}:
        raise ProtocolError('UCR 2018 is a univariate archive')
    if profile.task == 'ts_regression':
        train_y, test_y = np.asarray(train_y, dtype=float), np.asarray(test_y, dtype=float)
        if not (np.isfinite(train_y).all() and np.isfinite(test_y).all()):
            raise ProtocolError('Regression targets must be finite numbers')
    else:
        if any(label is None or str(label).lower() in ('', 'nan', '?') for label in (*train_y, *test_y)):
            raise ProtocolError('Classification targets must be present')
        train_y, test_y = np.asarray(train_y, dtype=str), np.asarray(test_y, dtype=str)
    identity = json.dumps([profile.version, dataset, train.sha256, test.sha256], separators=(',', ':'))
    return ArchiveSplit(profile, dataset, train_cases, train_y, test_cases, test_y,
                        sha256(identity.encode('utf-8')).hexdigest())


def _euclidean_features(split):
    def signature(case):
        return tuple(len(channel) for channel in case)

    expected = signature(split.train[0])
    if not all(expected) or any(signature(case) != expected for case in (*split.train, *split.test)):
        raise ProtocolError('Euclidean 1-NN requires equal lengths per channel; no padding is applied')
    train = np.stack([np.concatenate(case) for case in split.train])
    test = np.stack([np.concatenate(case) for case in split.test])
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ProtocolError('Euclidean 1-NN requires observed finite values; no imputation is applied')
    return train, test


def _regression_features(cases):
    def statistics(channel):
        values = channel[np.isfinite(channel)]
        return (values.mean(), values.std()) if len(values) else (np.nan, np.nan)

    return np.asarray([[value for channel in case for value in statistics(channel)] for case in cases])


def evaluate_archive_reference(split: ArchiveSplit):
    """Run one fixed model using TRAIN only, then score the unchanged official TEST."""
    if split.profile.task == 'ts_classification':
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.neighbors import KNeighborsClassifier

        x_train, x_test = _euclidean_features(split)
        estimator = KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='euclidean', n_jobs=1)
        prediction = estimator.fit(x_train, split.train_target).predict(x_test)
        metric_values = {'accuracy': accuracy_score(split.test_target, prediction),
                         'balanced_accuracy': balanced_accuracy_score(split.test_target, prediction),
                         'f1_macro': f1_score(split.test_target, prediction, average='macro', zero_division=0)}
        model = '1nn_euclidean_raw@1'
    else:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge())
        prediction = estimator.fit(_regression_features(split.train), split.train_target).predict(
            _regression_features(split.test))
        metric_values = {'rmse': np.sqrt(np.mean((split.test_target - prediction) ** 2))}
        model = 'channel_mean_std_ridge@1'
    return {'release': split.profile.version, 'dataset': split.dataset, 'split_id': split.split_id, 'model': model,
            'metrics': {key: float(value) for key, value in metric_values.items()}}


def archive_coverage(profile, scope, outcomes):
    """Include every published task; distinguish not selected from failed and absent."""
    selected = set(profile.selection(scope))
    if set(outcomes) - selected:
        raise ProtocolError('Outcomes contain tasks outside the frozen selection')
    tasks = {
        task: ({'status': 'not_selected', 'reason': 'outside frozen smoke selection'} if task not in selected else
               outcomes.get(task, {'status': 'unavailable', 'reason': 'no outcome reported'}))
        for task in profile.tasks
    }
    successful = sum(result['status'] == 'success' for result in tasks.values())
    return {'release': profile.version, 'scope': scope, 'expected': len(tasks), 'successful': successful,
            'full_archive': scope == 'full' and successful == len(tasks), 'tasks': tasks}


def run_archive_reference(profile, catalog, root, *, scope):
    """Evaluate the frozen selection; never discover alternate files or download missing data."""
    outcomes = {}
    for dataset in profile.selection(scope):
        release = catalog.get(f'{profile.version}/{dataset}')
        if release is None or release.license is None:
            reason = 'dataset not in local catalog' if release is None else 'dataset license is unknown'
            outcomes[dataset] = {'status': 'unavailable', 'reason': reason}
            continue
        try:
            assets = dict(release.assets)
            if release.version != profile.version or not {'train', 'test'} <= assets.keys():
                raise ProtocolError('Catalog entry must match the release and name both split roles')
            split = load_archive_split(profile, dataset, root, assets['train'], assets['test'])
            outcomes[dataset] = {'status': 'success', **evaluate_archive_reference(split)}
        except AssetUnavailable as error:
            outcomes[dataset] = {'status': 'unavailable', 'reason': str(error)}
        except AssetCorrupt as error:
            outcomes[dataset] = {'status': 'corrupt', 'reason': str(error)}
        except (AssetIncompatible, ValueError) as error:
            outcomes[dataset] = {'status': 'incompatible', 'reason': str(error)}
    return archive_coverage(profile, scope, outcomes)


def compare_published_accuracy(result, reference):
    """Only compare claims with an explicit identical release and split, and accuracy."""
    keys = ('release', 'dataset', 'split_id')
    if any(not result.get(key) or result[key] != reference.get(key) for key in keys):
        raise ProtocolError('Reference release, dataset and split must match exactly')
    if reference.get('metric') != 'accuracy' or 'accuracy' not in result.get('metrics', {}):
        raise ProtocolError('Published accuracy can only be compared with accuracy')
    return result['metrics']['accuracy'] - float(reference['value'])


def audit_pems_labels(uci_train, uci_test, permutation, uea_train, uea_test):
    """Compare split labels and restore calendar order without conflating file variants."""
    uci = tuple(uci_train) + tuple(uci_test)
    permutation = tuple(permutation)
    if len(permutation) != len(uci) or sorted(permutation) != list(range(1, len(uci) + 1)):
        raise ProtocolError('UCI PEMS-SF randperm must be a 1-based permutation of all records')
    chronological = tuple(label for _, label in sorted(zip(permutation, uci, strict=True)))
    return {'source_variants': ('uci_pems_sf', 'uea_2018/PEMS-SF'), 'independent_datasets': 1,
            'uci_sizes': (len(uci_train), len(uci_test)), 'uea_sizes': (len(uea_train), len(uea_test)),
            'same_ordered_labels': tuple(uci_train) == tuple(uea_train) and tuple(uci_test) == tuple(uea_test),
            'uci_chronological_labels': chronological}
