from hashlib import sha256

import numpy as np
import pytest

from benchmark.industrial.datasets.archives import (
    archive_coverage,
    archive_profile,
    audit_pems_labels,
    compare_published_accuracy,
    evaluate_archive_reference,
    load_archive_split,
)
from benchmark.industrial.datasets.catalog import CATALOG_VERSION, parse_catalog
from benchmark.industrial.datasets.ltsf import load_ltsf_csv
from benchmark.industrial.datasets.m5 import load_m5_window, score_wrmsse
from benchmark.industrial.protocols.assets import AssetCorrupt, LocalAsset
from benchmark.industrial.protocols.schema import ProtocolError


def asset(root, name, text):
    data = text.encode('utf-8')
    (root / name).write_bytes(data)
    return LocalAsset(name, sha256(data).hexdigest())


def catalog(root, roles, *, license_id='CC0-1.0'):
    assets = {role: asset(root, name, body) for role, (name, body) in roles.items()}
    payload = {'version': CATALOG_VERSION, 'releases': {'fixture': {
        'version': '1', 'source': 'urn:synthetic', 'access': 'local', 'license': license_id,
        'assets': {role: vars(item) for role, item in assets.items()},
    }}}
    return parse_catalog(payload)['fixture']


@pytest.mark.parametrize('version,count,selected', [
    ('ucr_2018', 128, 'ECG200'), ('uea_2018', 30, 'BasicMotions'), ('tser_original_19', 19, 'AppliancesEnergy'),
])
def test_frozen_archive_coverage(version, count, selected):
    profile = archive_profile(version)
    assert profile.selection('smoke') == (selected,)
    assert len(set(profile.tasks)) == len(profile.tasks) == count
    smoke = archive_coverage(profile, 'smoke', {selected: {'status': 'success'}})
    assert not smoke['full_archive']
    assert smoke['successful'] == 1
    assert sum(row['status'] == 'not_selected' for row in smoke['tasks'].values()) == count - 1
    full = archive_coverage(profile, 'full', {selected: {'status': 'failed', 'reason': 'unavailable dependency'}})
    assert not full['full_archive']
    assert full['tasks'][selected]['reason'] == 'unavailable dependency'
    assert sum(row['status'] == 'unavailable' for row in full['tasks'].values()) == count - 1


def test_ucr_baseline_and_published_metric_guard(tmp_path):
    profile = archive_profile('ucr_2018')
    train = asset(tmp_path, 'ECG200_TRAIN.tsv', 'a\t0\t0\nb\t2\t2\n')
    test = asset(tmp_path, 'ECG200_TEST.tsv', 'a\t0\t1\nb\t2\t1\n')
    split = load_archive_split(profile, 'ECG200', tmp_path, train, test, verify_counts=False)
    scores = evaluate_archive_reference(split)
    assert scores['metrics'] == {'accuracy': 1., 'balanced_accuracy': 1., 'f1_macro': 1.}
    reference = {**scores, 'metric': 'accuracy', 'value': .5}
    assert compare_published_accuracy(scores, reference) == .5
    with pytest.raises(ProtocolError, match='only be compared'):
        compare_published_accuracy(scores, {**reference, 'metric': 'balanced_accuracy'})
    with pytest.raises(ProtocolError, match='split must match'):
        compare_published_accuracy(scores, {**reference, 'split_id': 'changed'})
    with pytest.raises(ProtocolError, match='paths'):
        load_archive_split(profile, 'ECG200', tmp_path, test, train, verify_counts=False)
    with pytest.raises(ProtocolError, match='sizes'):
        load_archive_split(profile, 'ECG200', tmp_path, train, test)
    (tmp_path / train.path).write_bytes(b'invalid')
    with pytest.raises(AssetCorrupt):
        load_archive_split(profile, 'ECG200', tmp_path, train, test, verify_counts=False)


def test_uea_preserves_channels_and_does_not_pad(tmp_path):
    profile = archive_profile('uea_2018')
    header = '@dimensions 2\n@classLabel true a b\n@data\n'
    train = asset(tmp_path, 'BasicMotions_TRAIN.ts', header + '0,0:2,2:a\n2,2:0,0:b\n')
    test = asset(tmp_path, 'BasicMotions_TEST.ts', header + '0:2,?:a\n')
    split = load_archive_split(profile, 'BasicMotions', tmp_path, train, test, verify_counts=False)
    assert [len(channel) for channel in split.test[0]] == [1, 2]
    assert np.isnan(split.test[0][1][1])
    with pytest.raises(ProtocolError, match='no padding'):
        evaluate_archive_reference(split)
    test = asset(tmp_path, 'BasicMotions_TEST.ts', header + '0,1:2,2:a\n')
    split = load_archive_split(profile, 'BasicMotions', tmp_path, train, test, verify_counts=False)
    assert evaluate_archive_reference(split)['metrics']['accuracy'] == 1


def test_tser_numeric_targets_and_fixed_training_baseline(tmp_path):
    profile = archive_profile('tser_original_19')
    header = '@timestamps true\n@dimensions 1\n@targetLabel true\n@data\n'
    train = asset(tmp_path, 'AppliancesEnergy_TRAIN.ts', header + '(1,1),(2,?):1\n(1,2),(2,2):3\n')
    test = asset(tmp_path, 'AppliancesEnergy_TEST.ts', header + '(1,1):2\n')
    split = load_archive_split(profile, 'AppliancesEnergy', tmp_path, train, test, verify_counts=False)
    assert split.train_target.tolist() == [1., 3.]
    assert evaluate_archive_reference(split)['metrics']['rmse'] == pytest.approx(2 / 3)
    test = asset(tmp_path, 'AppliancesEnergy_TEST.ts', '@classLabel true a\n@data\n1:a\n')
    with pytest.raises(ProtocolError, match='target kind'):
        load_archive_split(profile, 'AppliancesEnergy', tmp_path, train, test, verify_counts=False)


def test_pems_distinct_variants_one_source():
    audit = audit_pems_labels([2, 1], [3], [2, 1, 3], [2, 1], [3])
    assert audit['same_ordered_labels']
    assert audit['uci_chronological_labels'] == (1, 2, 3)
    assert audit['independent_datasets'] == 1
    with pytest.raises(ProtocolError, match='permutation'):
        audit_pems_labels([2, 1], [3], [2, 2, 3], [2, 1], [3])


def test_ltsf_catalog_and_causal_inputs(tmp_path):
    content = 'date,target,observed,known\n2024-01-01,1,10,0\n2024-01-02,2,20,1\n2024-01-03,3,30,2\n'
    release = catalog(tmp_path, {'series': ('series.csv', content)})
    table = load_ltsf_csv(release, tmp_path, targets=('target',), observed=('observed',), known_future=('known',))
    window = table.window(1, 1)
    assert window.inputs.history.ravel().tolist() == [1., 2.]
    assert window.inputs.observed.ravel().tolist() == [10., 20.]
    assert window.inputs.known_future.ravel().tolist() == [2.]
    assert window.truth.ravel().tolist() == [3.]
    assert window.inputs.history.base is None
    assert not window.inputs.history.flags.writeable
    with pytest.raises(ProtocolError, match='duplicates'):
        load_ltsf_csv(release, tmp_path, targets=('target',), known_future=('target',))
    (tmp_path / 'series.csv').write_bytes(b'bad')
    with pytest.raises(AssetCorrupt):
        load_ltsf_csv(release, tmp_path, targets=('target',))
    unknown = catalog(tmp_path, {'series': ('series.csv', content)}, license_id=None)
    with pytest.raises(ProtocolError, match='license is unknown'):
        load_ltsf_csv(unknown, tmp_path, targets=('target',))


def m5_catalog(root, *, constant=False):
    days = 58
    header = 'item_id,dept_id,cat_id,store_id,state_id,' + ','.join(f'd_{day}' for day in range(1, days + 1))
    first = ','.join(str(1 if constant else 0 if day == 1 else day - 1) for day in range(1, days + 1))
    sales = header + '\nitem,dept,cat,store,state,' + first + '\n'
    calendar = 'd,wm_yr_wk\n' + ''.join(f'd_{day},{1 if day <= 30 else 2}\n' for day in range(1, days + 1))
    prices = 'store_id,item_id,wm_yr_wk,sell_price\nstore,item,1,2\nstore,item,2,5\n'
    return catalog(root, {'sales': ('sales.csv', sales), 'calendar': ('calendar.csv', calendar),
                          'prices': ('prices.csv', prices)})


def test_m5_wrm_sse_and_available_prices(tmp_path):
    release = m5_catalog(tmp_path)
    window = load_m5_window(release, tmp_path, 30)
    assert window.history.shape == (1, 30)
    assert window.future_prices is None
    assert window.last_dollar_sales[0] == 2 * sum(range(2, 30))
    with_prices = load_m5_window(release, tmp_path, 30, future_prices_known=True)
    assert with_prices.future_prices.tolist() == [[5.] * 28]
    result = score_wrmsse(window, window.truth + 2, target_days=window.horizon, series_keys=window.series_keys)
    assert len(result['levels']) == 12
    assert result['wrmsse'] == 2
    with pytest.raises(ProtocolError, match='exact forecast days'):
        score_wrmsse(window, window.truth, target_days=window.horizon[::-1], series_keys=window.series_keys)
    with pytest.raises(ProtocolError, match='item/store order'):
        score_wrmsse(window, window.truth, target_days=window.horizon, series_keys=(('another item', 'store'),))


def test_m5_undefined_positive_weight_scale(tmp_path):
    window = load_m5_window(m5_catalog(tmp_path, constant=True), tmp_path, 30)
    with pytest.raises(ProtocolError, match='positive weight, zero scale'):
        score_wrmsse(window, window.truth, target_days=window.horizon, series_keys=window.series_keys)
