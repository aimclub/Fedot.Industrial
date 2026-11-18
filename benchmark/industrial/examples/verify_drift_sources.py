"""Check DATA-07 source contracts offline and print aggregate evidence only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.industrial.datasets.catalog import CatalogRelease
from benchmark.industrial.datasets.drift import load_hydraulic_cycles
from benchmark.industrial.protocols.assets import LocalAsset, checked_asset
from benchmark.industrial.protocols.json_io import read_protocol_json
from benchmark.industrial.protocols.schema import Access


def verify_hydraulic(root, config):
    assets = [('profile', LocalAsset('profile.txt', config['profile_sha256']))]
    assets.extend((sensor, LocalAsset(f'{sensor}.txt', spec['sha256'])) for sensor, spec in config['sensors'].items())
    release = CatalogRelease('uci_hydraulic', config['version'], config['source'], Access.LOCAL,
                             config['license'], tuple(assets))
    channels, target = load_hydraulic_cycles(release, root / 'hydraulic', sensors=tuple(config['sensors']),
                                             target_column=0)
    expected = {sensor: (config['expected_cycles'], config['cycle_seconds'] * spec['rate_hz'])
                for sensor, spec in config['sensors'].items()}
    shapes = {sensor: values.shape for sensor, values in channels.items()}
    if shapes != expected or not all(np.isfinite(values).all() for values in channels.values()):
        raise ValueError('Hydraulic sensor coverage, sampling rates or values differ from the UCI release')
    values, counts = np.unique(target, return_counts=True)
    counts = {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}
    if counts != config['cooler_counts']:
        raise ValueError('Hydraulic cooler class counts differ from the UCI release')
    return {'release': release.version, 'license': release.license, 'cycles': len(target),
            'sensors': shapes, 'cooler_counts': counts, 'split': 'no_published_trial_partition'}


def graphormer_label_audit(metadata, config):
    """Reconstruct the published J44 data path without loading signals or model code."""
    if (metadata.empty or metadata.isna().any().any() or not metadata['Patient_id'].is_unique
            or not pd.api.types.is_integer_dtype(metadata['Week'])
            or not pd.api.types.is_integer_dtype(metadata['D_class'])):
        raise ValueError('The pinned S-OH metadata requires unique, complete records and integer codes')
    cohort = metadata.loc[(metadata.Week <= config['week_max']) & (metadata.D_class <= config['class_max'])]
    train = cohort.loc[cohort.Week.isin(config['train_weeks'])]
    test = cohort.loc[cohort.Week.isin(config['test_weeks'])]
    overlap = set(train.Patient_id) & set(test.Patient_id)
    if overlap or train.empty or test.empty or len(train) + len(test) != len(cohort):
        raise ValueError('Published script weeks do not partition the declared cohort')

    code = config['target_code']
    diagnoses = test.D_class.to_numpy()
    # The script trains on 0 for the target, but the graph builder receives raw test codes.
    intended, actual = diagnoses != code, diagnoses != 0
    valid_count = int(len(train) * config['validation_fraction'])
    if not 0 < valid_count < len(train):
        raise ValueError('The validation portion must leave observations on both sides')
    validation = []
    classes = train.D_class.to_numpy()
    for seed in config['validation_seeds']:
        picked = np.random.RandomState(seed).choice(len(train), size=valid_count, replace=False)
        table_order = np.sort(picked)
        validation.append({
            'seed': seed, 'rows': valid_count,
            'positions_out_of_order': int(np.count_nonzero(picked != table_order)),
            'labels_changed': int(np.count_nonzero((classes[picked] != code) != (classes[table_order] != code))),
        })
    return {'metadata_rows': len(metadata), 'script_cohort': len(cohort), 'train_rows': len(train),
            'test_rows': len(test), 'overlapping_groups': len(overlap),
            'target_train': int(np.sum(train.D_class == code)), 'target_test': int(np.sum(diagnoses == code)),
            'healthy_test': int(np.sum(diagnoses == 0)), 'test_labels_changed': int(np.sum(intended != actual)),
            'validation': validation}


def verify_breath_dx(root, config):
    asset = config['metadata']
    path = checked_asset(root, LocalAsset(asset['path'], asset['sha256']))
    frame = pd.read_csv(path, usecols=['Patient_id', 'Week', 'D_class'])
    return {**graphormer_label_audit(frame, config), 'dataset_revision': config['dataset_revision'],
            'code_revision': config['code_revision'], 'reported_auc_status': config['score_status']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--suite', choices=('hydraulic', 'breath_dx', 'all'), default='all')
    args = parser.parse_args(argv)
    config = read_protocol_json(Path(__file__).with_name('drift_acceptance_sources.json'))
    if config['version'] != 'industrial_drift_acceptance@1':
        raise ValueError('Unrecognized DATA-07 manifest')
    runners = {'hydraulic': verify_hydraulic, 'breath_dx': verify_breath_dx}
    selected = runners if args.suite == 'all' else (args.suite,)
    print(json.dumps({name: runners[name](args.data_root, config[name]) for name in selected},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
