from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.industrial.datasets.archives import (
    archive_coverage,
    archive_profile,
    audit_pems_labels,
    compare_published_accuracy,
    evaluate_archive_reference,
    load_archive_split,
    run_archive_reference,
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
    if version == 'ucr_2018':
        assert 'Yoga' in profile.tasks and 'AsphaltObstacles' not in profile.tasks
    assert selected in dict(profile.verified_splits)
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
    assert scores['official_split'] is False
    reference = {**scores, 'metric': 'accuracy', 'value': .5}
    with pytest.raises(ProtocolError, match='verified official split'):
        compare_published_accuracy(scores, reference)
    trusted = replace(profile, verified_splits=(('ECG200', (train.sha256, test.sha256)),))
    verified = load_archive_split(trusted, 'ECG200', tmp_path, train, test, verify_counts=False)
    scores = evaluate_archive_reference(verified)
    assert scores['official_split'] and scores['split_id'] == reference['split_id']
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


def test_user_supplied_hashes_do_not_certify_official_archive_partitions(tmp_path):
    profile = archive_profile('ucr_2018')
    train = 'a\t0\nb\t2\n' * 50
    test = 'a\t1\nb\t3\n' * 50
    entry = catalog(tmp_path, {'train': ('ECG200_TRAIN.tsv', train), 'test': ('ECG200_TEST.tsv', test)})
    key = 'ucr_2018/ECG200'
    entry = replace(entry, release_id=key, version=profile.version)
    report = run_archive_reference(profile, {key: entry}, tmp_path, scope='smoke')
    assert report['successful'] == 0
    assert report['tasks']['ECG200']['status'] == 'unverified_split'
    assert 'metrics' not in report['tasks']['ECG200']
    assets = dict(entry.assets)
    verified = replace(profile, verified_splits=(('ECG200', (assets['train'].sha256, assets['test'].sha256)),))
    report = run_archive_reference(verified, {key: entry}, tmp_path, scope='smoke')
    assert report['successful'] == 1 and report['tasks']['ECG200']['official_split']
    (tmp_path / assets['test'].path).unlink()
    report = run_archive_reference(verified, {key: entry}, tmp_path, scope='smoke')
    assert report['tasks']['ECG200']['status'] == 'unavailable'


def test_archive_parses_verified_snapshot_not_a_second_open(tmp_path, monkeypatch):
    from benchmark.industrial.datasets import archives

    profile = archive_profile('ucr_2018')
    train = asset(tmp_path, 'ECG200_TRAIN.tsv', 'a\t0\nb\t2\n')
    test = asset(tmp_path, 'ECG200_TEST.tsv', 'a\t1\nb\t3\n')
    read = archives.read_checked_asset

    def replace_after_read(root, value):
        path, content = read(root, value)
        path.write_text('unrelated content', encoding='utf-8')
        return path, content

    monkeypatch.setattr(archives, 'read_checked_asset', replace_after_read)
    split = load_archive_split(profile, 'ECG200', tmp_path, train, test, verify_counts=False)
    assert split.test_target.tolist() == ['a', 'b']
    assert split.test[0][0].tolist() == [1.]
    assert not split.test_target.flags.writeable and not split.test[0][0].flags.writeable


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


@pytest.mark.parametrize('content,message', [
    ('date,target,target\n2024-01-01,1,2\n', 'duplicates'),
    ('date,target\n20240101,1\n', 'not numeric epochs'),
])
def test_ltsf_rejects_ambiguous_columns_and_times(tmp_path, content, message):
    release = catalog(tmp_path, {'series': ('series.csv', content)})
    with pytest.raises(ProtocolError, match=message):
        load_ltsf_csv(release, tmp_path, targets=('target',))


def test_ltsf_parses_the_csv_bytes_that_were_verified(tmp_path, monkeypatch):
    original = 'date,target\n2024-01-01,1\n2024-01-02,2\n'
    changed = original.replace(',2\n', ',9\n')
    release = catalog(tmp_path, {'series': ('series.csv', original)})
    source = tmp_path / 'series.csv'
    read_bytes = Path.read_bytes

    def replace_after_read(path):
        content = read_bytes(path)
        if path == source:
            path.write_text(changed, encoding='utf-8')
        return content

    monkeypatch.setattr(Path, 'read_bytes', replace_after_read)
    result = load_ltsf_csv(release, tmp_path, targets=('target',))
    assert source.read_text(encoding='utf-8') == changed
    assert result.target[:, 0].tolist() == [1., 2.]


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


def test_m5_parses_the_verified_sales_snapshot(tmp_path, monkeypatch):
    release = m5_catalog(tmp_path)
    source = tmp_path / 'sales.csv'
    original = source.read_text(encoding='utf-8')
    changed = original.replace('item,dept,cat,store,state,0,1,', 'item,dept,cat,store,state,9,1,')
    read_csv = pd.read_csv

    def replace_before_read(table, *args, **kwargs):
        source.write_text(changed, encoding='utf-8')
        return read_csv(table, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', replace_before_read)
    window = load_m5_window(release, tmp_path, 30)
    assert source.read_text(encoding='utf-8') == changed
    assert window.history[0, 0] == 0.


def test_m5_undefined_positive_weight_scale(tmp_path):
    window = load_m5_window(m5_catalog(tmp_path, constant=True), tmp_path, 30)
    result = score_wrmsse(window, window.truth, target_days=window.horizon, series_keys=window.series_keys)
    assert result == {'protocol': 'm5_wrmsse@1', 'status': 'not_available', 'reason': 'zero_scale: total',
                      'wrmsse': None, 'levels': {}}
    missing = replace(window, truth=None)
    outcome = score_wrmsse(missing, window.truth, target_days=window.horizon, series_keys=window.series_keys)
    assert outcome['reason'] == 'missing_truth' and outcome['wrmsse'] is None


def test_m5_organizer_calendar_and_separate_targets(tmp_path):
    import pandas as pd

    m5_catalog(tmp_path)
    sales = pd.read_csv(tmp_path / 'sales.csv')
    sales = pd.concat([sales, sales.assign(item_id='item2')], ignore_index=True)
    past = list(sales.columns[:35])
    days = list(sales.columns[35:])
    truth = sales.drop(columns=past[5:]).iloc[::-1].copy()
    truth.loc[truth['item_id'] == 'item2', days] += 10
    calendar = pd.read_csv(tmp_path / 'calendar.csv').drop(columns='d')
    calendar['date'] = pd.date_range('2011-01-29', periods=len(calendar)).strftime('%Y-%m-%d')
    prices = (tmp_path / 'prices.csv').read_text()
    price_rows = prices.splitlines()[1:]
    prices += ''.join(f'{row.replace(",item,", ",item2,")}\n' for row in price_rows)

    def release(targets, dates):
        return catalog(tmp_path, {'sales': ('sales.csv', sales[past].to_csv(index=False)),
                                  'calendar': ('calendar.csv', dates.to_csv(index=False)),
                                  'prices': ('prices.csv', prices),
                                  'targets': ('targets.csv', targets.to_csv(index=False))})

    window = load_m5_window(release(truth, calendar), tmp_path, 30)
    assert window.series_keys == (('item', 'store'), ('item2', 'store'))
    identifiers = window.identifiers
    identifiers.loc[0, 'item_id'] = 'other'
    assert window.series_keys == (('item', 'store'), ('item2', 'store'))
    assert window.truth[:, 0].tolist() == [30, 40]
    assert window.future_calendar.iloc[0]['date'] == '2011-02-28'
    with pytest.raises(ProtocolError, match='do not match'):
        load_m5_window(release(truth.iloc[:1], calendar), tmp_path, 30)
    with pytest.raises(ProtocolError, match='duplicate'):
        load_m5_window(release(pd.concat([truth, truth]), calendar), tmp_path, 30)
    with pytest.raises(ProtocolError, match='consecutive dates'):
        load_m5_window(release(truth, calendar.iloc[1:]), tmp_path, 30)
    bad_calendar = calendar.assign(d=[f'd_{day}' for day in range(2, len(calendar) + 2)])
    with pytest.raises(ProtocolError, match='IDs do not match'):
        load_m5_window(release(truth, bad_calendar), tmp_path, 30)


def test_hydraulic_loader_preserves_each_sensor_grid_and_cycle_labels(tmp_path):
    from benchmark.industrial.datasets.drift import load_hydraulic_cycles

    payload = {'profile': ('profile.txt', '3\t0\n100\t1\n'), 'PS1': ('PS1.txt', '1\t2\t3\n4\t5\t6\n'),
               'TS1': ('TS1.txt', '10\n11\n')}
    release = catalog(tmp_path, payload)
    channels, targets = load_hydraulic_cycles(release, tmp_path, sensors=('PS1', 'TS1'), target_column=0)
    assert channels['PS1'].shape == (2, 3) and channels['TS1'].shape == (2, 1)
    assert targets.tolist() == [3, 100]
    malformed = catalog(tmp_path, {**payload, 'TS1': ('TS1.txt', '10\n')})
    with pytest.raises(ProtocolError, match='cycle level'):
        load_hydraulic_cycles(malformed, tmp_path, sensors=('PS1', 'TS1'), target_column=0)


def test_hydraulic_loader_parses_the_verified_profile_snapshot(tmp_path, monkeypatch):
    from benchmark.industrial.datasets.drift import load_hydraulic_cycles

    release = catalog(tmp_path, {'profile': ('profile.txt', '3\n100\n'), 'TS1': ('TS1.txt', '10\n11\n')})
    source = tmp_path / 'profile.txt'
    loadtxt = np.loadtxt

    def replace_before_read(table, *args, **kwargs):
        source.write_text('20\n100\n', encoding='utf-8')
        return loadtxt(table, *args, **kwargs)

    monkeypatch.setattr(np, 'loadtxt', replace_before_read)
    _, targets = load_hydraulic_cycles(release, tmp_path, sensors=('TS1',), target_column=0)
    assert source.read_text(encoding='utf-8').splitlines() == ['20', '100']
    assert targets.tolist() == [3., 100.]


@pytest.mark.parametrize('code', ['3.5', 'nan', 'inf'])
def test_hydraulic_acceptance_rejects_invalid_cooler_codes(tmp_path, code):
    from benchmark.industrial.examples.verify_drift_sources import verify_hydraulic

    root = tmp_path / 'hydraulic'
    root.mkdir()
    profile = asset(root, 'profile.txt', f'{code}\n20\n')
    sensor = asset(root, 'TS1.txt', '10\n11\n')
    config = {'version': 'fixture', 'source': 'urn:synthetic', 'license': 'CC0-1.0',
              'profile_sha256': profile.sha256, 'expected_cycles': 2, 'cycle_seconds': 1,
              'sensors': {'TS1': {'sha256': sensor.sha256, 'rate_hz': 1}}, 'cooler_counts': {'3': 1, '20': 1}}
    with pytest.raises(ValueError, match='cooler codes must be finite integers'):
        verify_hydraulic(tmp_path, config)
