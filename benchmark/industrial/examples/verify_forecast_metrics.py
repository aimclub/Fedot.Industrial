"""Reproduce DATA-02/05 acceptance on fixed, local forecast files without training."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from math import fsum
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from benchmark.industrial.datasets.catalog import CatalogRelease
from benchmark.industrial.datasets.m5 import load_m5_window, score_wrmsse
from benchmark.industrial.evaluation.forecasting_protocols import (
    ForecastEvidence, ForecastModelScope, m4_naive2, score_m4, score_monash,
)
from benchmark.industrial.protocols.assets import LocalAsset, checked_asset
from benchmark.industrial.protocols.json_io import read_protocol_json
from benchmark.industrial.protocols.schema import Access


_M5_ROLES = {
    'sales': 'm5/sales_train_evaluation.csv', 'targets': 'm5/sales_test_evaluation.csv',
    'calendar': 'm5/calendar.csv', 'prices': 'm5/sell_prices.csv',
}
_M5_COUNTS = (1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490)
_SHEET_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def _m5_published_scores(path):
    # The pinned workbook's third sheet is Accuracy-Top50 (AL).
    ns = _SHEET_NS
    with ZipFile(path) as book:
        strings = ElementTree.fromstring(book.read('xl/sharedStrings.xml'))
        sheet = ElementTree.fromstring(book.read('xl/worksheets/sheet3.xml'))
    row = sheet.find(f"{ns}sheetData/{ns}row[@r='3']")
    cells = {cell.attrib['r']: cell.findtext(f'{ns}v') for cell in row}
    team = ''.join(node.text or '' for node in strings[int(cells['A3'])].iter(f'{ns}t'))
    if team != 'YJ_STU':
        raise ValueError('YJ_STU is not in the expected leaderboard row')
    return [float(cells[f'{column}3']) for column in 'BCDEFGHIJKLMN']


def _m5_reference(window, forecast, published_weights):
    from scipy.sparse import csr_matrix

    # Level11 is state/item in the organizer file, and item/state in Industrial.
    state, store, category, dept, item = (window.identifiers[key].to_numpy()
                                          for key in ('state_id', 'store_id', 'cat_id', 'dept_id', 'item_id'))
    x = np.repeat('X', len(item))
    label_pairs = ((np.repeat('Total', len(item)), x), (state, x), (store, x), (category, x), (dept, x),
                   (state, category), (state, dept), (store, category), (store, dept), (item, x),
                   (state, item), (item, store))
    residual = window.truth - forecast
    scores, weight_gaps, revenue_gaps = [], [], []
    for number, (first, second) in enumerate(label_pairs, 1):
        official = published_weights.loc[published_weights['Level_id'] == f'Level{number}']
        keys = pd.MultiIndex.from_frame(official[['Agg_Level_1', 'Agg_Level_2']])
        groups = keys.get_indexer(pd.MultiIndex.from_arrays([first, second]))
        if len(keys) != _M5_COUNTS[number - 1] or set(groups) != set(range(len(keys))):
            raise ValueError(f'Unexpected M5 hierarchy coverage at level {number}')
        selector = csr_matrix((np.ones(len(item)), (groups, np.arange(len(item)))), shape=(len(keys), len(item)))
        reported = official['weight'].to_numpy()
        revenue = selector @ window.last_dollar_sales
        weight_gaps.append(float(np.max(np.abs(revenue / revenue.sum() - reported))))
        revenue_gaps.append(float(np.max(np.abs(revenue - official['Dollar_Sales'].to_numpy()))))

        # Match the official order: aggregate the forecasts first, then score the resulting series.
        history = selector @ window.history
        mse = np.mean((selector @ residual) ** 2, axis=1)
        positive = np.flatnonzero(reported > 0)
        rmsse = []
        for index in positive:
            series = history[index]
            start = np.flatnonzero(series > 0)[0]
            scale = np.mean(np.square(np.ediff1d(series[start:])))
            rmsse.append(float(np.sqrt(mse[index] / scale)))
        scores.append(fsum(np.asarray(rmsse) * reported[positive]))
    return scores, weight_gaps, revenue_gaps


def verify_m5(root, assets):
    roles = tuple((name, LocalAsset(path, assets[path])) for name, path in _M5_ROLES.items())
    release = CatalogRelease('m5_accuracy', 'evaluation', 'M5-methods', Access.LOCAL, 'M5 competition terms', roles)
    window = load_m5_window(release, root, 1941)
    submission = pd.read_csv(root / 'm5/0001 YJ_STU.csv').set_index('id', verify_integrity=True)
    keys = [f'{item}_{store}_evaluation' for item, store in window.series_keys]
    ids = submission.index[submission.index.str.endswith('_evaluation')]
    if len(keys) != 30490 or set(keys) != set(ids):
        raise ValueError('M5 requires the complete evaluation submission')
    forecast = submission.loc[keys, [f'F{k}' for k in range(1, 29)]].to_numpy(dtype=float)
    observed = score_wrmsse(window, forecast, target_days=window.horizon, series_keys=window.series_keys)
    published = _m5_published_scores(root / 'm5-scores.xlsx')
    reference, weights, revenue = _m5_reference(window, forecast, pd.read_csv(root / 'm5/weights_evaluation.csv'))
    actual = list(observed['levels'].values())
    np.testing.assert_allclose(revenue, 0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(weights, 0, rtol=0, atol=1e-11)
    np.testing.assert_allclose([*actual, observed['wrmsse']], [*reference, fsum(reference) / 12], rtol=0, atol=1e-9)
    np.testing.assert_allclose([*actual, observed['wrmsse']], published, rtol=0, atol=1e-8)
    return {'dataset': 'M5 evaluation', 'submission': 'YJ_STU', 'train_end': 'd_1941', 'horizon': 28,
            'levels': [{'level': index, 'series': count, 'industrial': a, 'independent': b, 'published': c}
                       for index, (count, a, b, c) in enumerate(zip(_M5_COUNTS, actual, reference, published), 1)],
            'wrmsse': observed['wrmsse'], 'published_wrmsse': published[-1],
            'max_weight_gap': max(weights), 'max_revenue_gap': max(revenue),
            'max_independent_level_gap': float(np.max(np.abs(np.array(actual) - reference))),
            'max_level_gap': float(np.max(np.abs(np.array(actual) - published[:-1])))}


def _m4_scores(rows, period, prediction):
    # Reduce independently (one scalar expression per observation, with math.fsum).
    smape, mase = [], []
    for row in rows:
        actual, predicted = row.truth, prediction(row)
        errors = [abs(float(a) - float(b)) for a, b in zip(actual, predicted, strict=True)]
        scale = (fsum(abs(a - b) for a, b in zip(row.history[period:], row.history[:-period], strict=True))
                 / (len(row.history) - period))
        smape.append(fsum(200 * e / (abs(a) + abs(b))
                          for e, a, b in zip(errors, actual, predicted, strict=True)) / len(errors))
        mase.append(fsum(errors) / len(errors) / scale)
    return fsum(smape) / len(rows), fsum(mase) / len(rows)


def verify_m4(root):
    train, truth, naive2, seasonal = (pd.read_csv(root / file, index_col=0) for file in (
        'm4-hourly-train.csv', 'm4-hourly-test.csv', 'submission-Naive2.csv', 'submission-sNaive.csv'))
    if len(train) != 414 or not train.index.is_unique or set(train.index) != set(truth.index):
        raise ValueError('Unexpected coverage in the official M4 Hourly split')
    grid = tuple(f'h+{index}' for index in range(1, 49))
    rows = [ForecastEvidence(key, train.loc[key].dropna().to_numpy(), truth.loc[key].to_numpy(),
                             seasonal.loc[key].to_numpy(), grid, grid, naive2.loc[key].to_numpy())
            for key in train.index]
    smape, mase = _m4_scores(rows, 24, lambda row: row.prediction)
    naive_smape, naive_mase = _m4_scores(rows, 24, lambda row: row.organizer_naive2)
    reference = (smape, mase, (smape / naive_smape + mase / naive_mase) / 2)
    observed = score_m4('Hourly', rows)
    np.testing.assert_allclose([observed[key] for key in ('smape', 'mase', 'owa')], reference, rtol=0, atol=1e-12)
    actual = np.stack([m4_naive2(row.history, 48, 24) for row in rows])
    expected = np.stack([row.organizer_naive2 for row in rows])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5.1e-5)
    reproduced = score_m4('Hourly', [replace(row, organizer_naive2=None) for row in rows])
    np.testing.assert_allclose(reproduced['owa'], observed['owa'], rtol=0, atol=1e-8)
    return {'dataset': 'M4 Hourly', 'submission': 'sNaive', **observed,
            'reference': dict(zip(('smape', 'mase', 'owa'), reference, strict=True)),
            'organizer_naive2_smape': naive_smape, 'organizer_naive2_mase': naive_mase,
            'reproduced_owa': reproduced['owa'],
            'naive2_max_gap': float(np.max(np.abs(actual - expected))),
            'nonconstant_naive2_series': int(np.count_nonzero(np.ptp(expected, axis=1)))}


def verify_monash(root):
    from sktime.datasets import load_tsf_to_dataframe
    from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

    frame, metadata = load_tsf_to_dataframe(str(root / 'tourism_monthly_dataset.tsf'), return_type='default_tsf')
    horizon = metadata['forecast_horizon']
    if len(frame) != 366 or horizon != 24 or metadata['frequency'] != 'monthly':
        raise ValueError('Unexpected Monash Tourism Monthly split')
    grid = tuple(f'month+{index}' for index in range(1, horizon + 1))
    rows, per_series = [], []
    for name, values in zip(frame['series_name'], frame['series_value'], strict=True):
        values = np.asarray(values, dtype=float)
        train, truth = values[:-horizon], values[-horizon:]
        # Fixed seasonal-repeat forecast using TRAIN only, not a fitted model.
        forecast = np.tile(train[-12:], 2)
        rows.append(ForecastEvidence(name, train, truth, forecast, grid, grid))
        per_series.append(mean_absolute_scaled_error(truth, forecast, y_train=train, sp=12))
    observed = score_monash(rows, (12,), scope=ForecastModelScope.LOCAL)
    reference = {'mean_mase': fsum(per_series) / len(per_series), 'median_mase': float(np.median(per_series))}
    np.testing.assert_allclose([observed[k] for k in reference], tuple(reference.values()), rtol=0, atol=1e-12)
    return {'dataset': 'Monash Tourism Monthly', 'forecast': 'seasonal_repeat@12', 'horizon': horizon,
            **observed, 'reference': reference}


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--suite', choices=('m5', 'm4', 'monash', 'all'), default='all')
    args = parser.parse_args(argv)
    manifest = read_protocol_json(Path(__file__).with_name('forecast_metric_acceptance_sources.json'))
    runners = {'m5': lambda: verify_m5(args.data_root, manifest['m5']),
               'm4': lambda: verify_m4(args.data_root), 'monash': lambda: verify_monash(args.data_root)}
    results = {}
    for name in runners if args.suite == 'all' else (args.suite,):
        assets = manifest[name]
        for path, digest in assets.items():
            checked_asset(args.data_root, LocalAsset(path, digest))
        results[name] = {'sha256': assets, **runners[name]()}
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
