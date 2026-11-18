from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial.datasets.catalog import parse_catalog, read_catalog
from benchmark.industrial.datasets.drift import (
    DriftCase, binary_drift_report, chronological_drift_split, overlapping_segments, published_drift_split,
)
from benchmark.industrial.datasets.ltsf import LTSFTable
from benchmark.industrial.evaluation.decomposition_protocol import (
    SyntheticSpec, component_errors, decomposition_control, seasonal_control, synthetic_case,
)
from benchmark.industrial.evaluation.detection_protocols import (
    EventWindow, TSB_AD_REVISION, average_precision, score_events, score_point_mask, score_tsb_ad, threshold_from_train,
)
from benchmark.industrial.evaluation.forecast_transfer import TrainingSources, evaluate_zero_shot, load_time_windows
from benchmark.industrial.evaluation.forecasting_protocols import (
    ForecastEvidence, ForecastModelScope, collection_identity, m4_naive2, reference_mase,
    reference_smape, score_m4, score_monash, seasonality_lag, unique_forecast_collections,
)
from benchmark.industrial.evaluation.protocol_report import pairwise_units, report_from_evidence
from benchmark.industrial.protocols import ProtocolError, resolve_protocol
from benchmark.industrial.protocols.__main__ import main
from benchmark.industrial.protocols.assets import file_digest


def _forecast(item, step, truth, prediction, reference):
    grid = tuple(f'day_{index}' for index in range(6))
    return ForecastEvidence(item, [1, 1 + step, 1 + 2 * step], [truth] * 6, [prediction] * 6, grid, grid,
                            [reference] * 6)


def test_m4_uses_aggregate_owa_and_original_horizons():
    rows = [_forecast('a', 1, 10, 8, 9), _forecast('b', 2, 20, 18, 16)]
    result = score_m4('Yearly', rows)
    smape = (200 * 2 / 18 + 200 * 2 / 38) / 2
    naive_smape = (200 / 19 + 200 * 4 / 36) / 2
    assert result['owa'] == pytest.approx((smape / naive_smape + 1.5 / 1.5) / 2)
    assert result['naive2_source'] == 'organizers'
    with pytest.raises(ProtocolError, match='horizon'):
        score_m4('Daily', rows)
    with pytest.raises(ProtocolError, match='mix'):
        score_m4('Yearly', [rows[0], replace(rows[1], organizer_naive2=None)])
    with pytest.raises(ProtocolError, match='aligned'):
        score_m4('Yearly', [replace(rows[0], prediction_times=rows[0].target_times[::-1])])


def test_forecast_scale_and_seasonal_controls():
    np.testing.assert_allclose(m4_naive2([1, 3, 7], 3, 1), [7, 7, 7])
    np.testing.assert_allclose(m4_naive2(np.tile([1, 3, 5, 2], 20), 5, 4), [1, 3, 5, 2, 1])
    assert reference_smape([0, 1, -1], [0, 0, 0]) == pytest.approx(400 / 3)
    with pytest.raises(ProtocolError, match='zero'):
        reference_mase([2, 2, 2], [3], [2], 1)
    with pytest.raises(ProtocolError, match='longer'):
        reference_mase([2], [3], [2], 1)
    assert seasonality_lag(([0] * 8, [0] * 10), (7, 365)) == 7
    assert seasonality_lag(([0] * 6, [0] * 10), (7, 365)) == 1
    assert score_monash([_forecast('a', 2, 10, 8, 9)], (7, 365), scope=ForecastModelScope.LOCAL)['mean_mase'] == 1
    assert collection_identity('monash', 'M4_Daily_dataset') == collection_identity('m4', 'Daily')
    with pytest.raises(ProtocolError, match='aliases'):
        unique_forecast_collections((('monash', 'M4_Daily_dataset'), ('m4', 'Daily')))


def test_detection_without_point_adjustment_and_with_event_delay():
    threshold = threshold_from_train([.1, .2, .3, .4], quantile=.75, direction='higher')
    assert threshold == pytest.approx(.325)
    labels = [0, 1, 1, 1, 0, 0]
    scores = [.1, .1, .9, .1, .1, .6]
    result = score_point_mask(scores, labels, threshold=threshold, direction='higher')
    assert result == {'tp': 1, 'fp': 1, 'fn': 2, 'f1': .4}
    events = score_events([EventWindow(0, 0), EventWindow(1, 3)], [2, 3, 5], length=6, sample_seconds=30)
    assert events == {'found': 1, 'missed': 1, 'delay_seconds': (30,), 'false_alarms': 1,
                      'repeated_alarms': 1, 'false_alarms_per_hour': 20}
    assert average_precision(scores, [0] * 6, direction='higher')['status'] == 'not_available'
    assert average_precision([-score for score in scores], labels, direction='lower') == average_precision(
        scores, labels, direction='higher')


def _tsb_scope(reference):
    return {'subset': 'TSB-AD-U', 'direction': 'higher', 'window': 2, 'thresholds': 8,
            'reference': reference, 'reference_revision': TSB_AD_REVISION}


def test_tsb_delegates_its_range_metric_and_reports_missing_reference():
    def reference(labels, scores, window, version, thresholds):
        assert (window, version, thresholds) == (2, 'opt', 8)
        np.testing.assert_array_equal(labels, [0, 1, 1, 0])
        assert np.argmax(scores) == 1
        return [0, .625]

    assert score_tsb_ad([.1, .9, .8, .1], [0, 1, 1, 0], **_tsb_scope(reference))['vus_pr'] == .625
    assert score_tsb_ad([.1, .9], [0, 1], **_tsb_scope(None))['status'] == 'not_available'
    with pytest.raises(ProtocolError, match='revision'):
        score_tsb_ad([.1, .9], [0, 1], **{**_tsb_scope(reference), 'reference_revision': 'main'})


def test_tsb_range_reference_when_installed():
    upstream = pytest.importorskip('TSB_AD.evaluation.basic_metrics')
    scores = [.1, .2, .4, .95, .85, .7, .15, .3]
    labels = [0, 0, 0, 1, 1, 1, 0, 0]
    assert score_tsb_ad(scores, labels, **_tsb_scope(upstream.generate_curve))['vus_pr'] == pytest.approx(1)


def test_tsb_declines_undefined_labels_and_missing_scores():
    def unused(*args):
        pytest.fail('VUS must not use invalid inputs')

    assert score_tsb_ad([.5, .9], [0, 0], **_tsb_scope(unused))['status'] == 'not_available'
    with pytest.raises(ProtocolError, match='finite'):
        score_tsb_ad([.5, np.nan], [0, 1], **_tsb_scope(unused))


def test_nab_elapsed_slots_keep_missing_intervals_in_delay_and_rate():
    import pandas as pd

    from benchmark.industrial.examples.verify_detection_metrics import _elapsed_slots, _timestamp_reference

    start = pd.Timestamp('2020-01-01')
    hour = pd.Timedelta(hours=1)
    windows = [(start + hour, start + 5 * hour)]
    alarms = [start, start + 4 * hour, start + 5 * hour]
    last = start + 5 * hour
    first, delay, false, repeats, exposure = _timestamp_reference(windows, alarms, start, last, hour)
    assert first == (start + 4 * hour,) and delay == (10_800,)
    assert (false, repeats, exposure) == (1, 1, 6)
    result = score_events([EventWindow(*_elapsed_slots(windows[0], start, hour))],
                          _elapsed_slots(alarms, start, hour), length=6, sample_seconds=3600)
    assert result['delay_seconds'] == delay and result['false_alarms_per_hour'] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match='time grid'):
        _elapsed_slots([start + hour / 2], start, hour)


def _drift(case, segment, target, start, end, *, group=None):
    return DriftCase.parse({'case_id': case, 'group_id': group or case, 'segment': segment, 'target': target,
                            'window_start': start, 'window_end': end, 'decision_time': end, 'event_time': end + 1})


def test_published_drift_not_assumed_chronological_and_never_emits_case_records():
    cases = [_drift('x', 'road1', 'negative', 4, 10), _drift('y', 'road1', 'positive', 1, 5),
             _drift('z', 'road2', 'negative', 2, 8)]
    parts = {'x': 'train', 'y': 'test', 'z': 'test'}
    split = published_drift_split(cases, parts, early_warning=True)
    assert overlapping_segments(split) == 1
    report = binary_drift_report(split, {'y': .9, 'z': .2}, positive='positive', negative='negative')
    assert report['roc_auc'] == 1 and report['test_count'] == 2
    assert set(report).isdisjoint({'x', 'y', 'z', 'road1', 'road2'})
    with pytest.raises(ProtocolError, match='partitions'):
        published_drift_split([cases[0], replace(cases[1], group_id='x'), cases[2]], parts)
    with pytest.raises(ProtocolError, match='cutoff'):
        chronological_drift_split(cases, 6)
    with pytest.raises(ProtocolError, match='precede'):
        published_drift_split([cases[0], replace(cases[1], event_time=5), cases[2]], parts, early_warning=True)


def test_graphormer_audit_distinguishes_disease_target_from_healthy_and_validation_order():
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from benchmark.industrial.examples.verify_drift_sources import graphormer_label_audit

    metadata = pd.DataFrame({'Patient_id': range(7), 'Week': [1] * 4 + [2] * 3, 'D_class': [7, 0, 7, 1, 7, 0, 1]})
    config = {'week_max': 2, 'class_max': 7, 'target_code': 7, 'train_weeks': (1,), 'test_weeks': (2,),
              'validation_fraction': .5, 'validation_seeds': (1,)}
    report = graphormer_label_audit(metadata, config)
    assert report['test_labels_changed'] == 2 and report['overlapping_groups'] == 0
    assert report['validation'] == [{'seed': 1, 'rows': 2, 'positions_out_of_order': 2, 'labels_changed': 2}]
    scores_for_not_target = [.1, .9, .9]
    assert roc_auc_score([0, 1, 1], scores_for_not_target) == 1
    assert roc_auc_score([1, 0, 1], scores_for_not_target) == .25
    assert 'Patient_id' not in json.dumps(report)
    with pytest.raises(ValueError, match='partition'):
        graphormer_label_audit(metadata, {**config, 'test_weeks': (1, 2)})


def test_zero_shot_has_no_future_label_or_unavailable_covariate():
    past = np.arange(5, dtype=float).reshape(-1, 1)
    dates = ('1', '2', '3', '4', '5')
    original = LTSFTable(dates, past, past, past)
    changed = past.copy()
    changed[3:] = 500
    shifted = LTSFTable(dates, changed, changed, past)
    seen = []

    def predict(inputs):
        assert not hasattr(inputs, 'truth')
        assert inputs.history.base is None and inputs.observed.base is None
        seen.append((inputs.history.tolist(), inputs.observed.tolist(), inputs.known_future.tolist()))
        return [[2], [2]]

    profile = dict(training=TrainingSources('model@1', ('m4/daily',), False),
                   dataset_aliases=('monash/m4_daily',), dates_public=True, frequency='D')
    first, second = [evaluate_zero_shot(predict, table.window(2, 2), **profile) for table in (original, shifted)]
    assert first['overlap'] == 'unknown' and first['mae'] != second['mae'] and seen[0] == seen[1]
    assert evaluate_zero_shot(predict, original.window(2, 2), input_layout='asynchronous',
                              **profile)['status'] == 'not_available'
    assert profile['training'].overlap(('m4/daily',)) == 'known_overlap'


def test_time_pinned_release_and_relative_test_tail(tmp_path):
    source = read_catalog(_EXAMPLE.with_name('time_transfer_smoke.json'))
    assert len(source) == 1 and next(iter(source.values())).license == 'Apache-2.0'
    path = tmp_path / 'time.csv'
    path.write_text('timestamp,y\n' + ''.join(f'2020-{month:02d}-01,{month}\n' for month in range(1, 8)),
                    encoding='utf-8')
    release = parse_catalog({'version': 'industrial_dataset_catalog@1', 'releases': {'mini': {
        'version': '1', 'source': 'urn:synthetic', 'access': 'local', 'license': 'CC0-1.0',
        'assets': {'series': {'path': path.name, 'sha256': file_digest(path)}},
    }}})['mini']
    windows = load_time_windows(release, tmp_path, target='y', frequency='MS', test_length=5, horizon=2)
    assert [w.inputs.history[-1, 0] for w in windows] == [2, 4]
    assert [w.truth[:, 0].tolist() for w in windows] == [[3, 4], [5, 6]]
    with pytest.raises(ProtocolError, match='frequency'):
        load_time_windows(release, tmp_path, target='y', frequency='D', test_length=5, horizon=2)


def test_synthetic_components_not_replaced_by_quality_of_sum():
    spec = SyntheticSpec.parse({'version': 'synthetic_decomposition@1', 'length': 64, 'horizon': 8,
                                'period': 8, 'train_seed': 42, 'evaluation_seed': 43})
    case, repeat = (synthetic_case(spec, 'evaluation') for _ in range(2))
    parts = case['components']
    np.testing.assert_array_equal(repeat['values'], case['values'])
    assert not np.array_equal(synthetic_case(spec, 'train')['values'], case['values'])
    assert all(row['rmse'] == 0 for row in component_errors(parts, parts).values())
    swapped = {**parts, 'trend': parts['seasonal'], 'seasonal': parts['trend']}
    np.testing.assert_allclose(sum(swapped.values()), case['values'])
    assert component_errors(parts, swapped)['trend']['rmse'] > 0
    shifted = {**parts, 'seasonal': np.roll(parts['seasonal'], 2)}
    assert component_errors(parts, shifted)['seasonal']['dominant_phase_error_radians'] == pytest.approx(np.pi / 2)
    assert seasonal_control(case)['mae'] > 0
    recovered = decomposition_control(case['values'], 8)
    np.testing.assert_allclose(sum(recovered.values()), case['values'])
    with pytest.raises(ProtocolError, match='every'):
        component_errors(parts, recovered)


_EXAMPLE = Path(__file__).resolve().parents[3] / 'benchmark/industrial/examples/protocol_smoke.json'


def _saved_report(task, metric_name):
    records = pytest.importorskip('roast.core.records')
    manifest = json.loads(_EXAMPLE.read_text(encoding='utf-8'))
    suite = manifest['suite']
    suite['task_kind'] = task
    suite['models'].append({**suite['models'][0], 'model_id': 'other'})
    suite['metrics'] = [{'schema_version': 1, 'metric_id': metric_name,
                         'metric': {'schema_version': 1, 'name': metric_name}}]
    suite['run']['primary_metric'] = metric_name
    profile = manifest['profiles']['toy']
    protocol = profile['protocol']
    protocol['mode'] = 'official'
    protocol['metrics'] = {metric_name: {'direction': 'minimize' if metric_name == 'rmse' else 'maximize',
                                         'version': '1', 'aggregation': 'mean_by_unit'}}
    protocol['split'] = {'kind': 'membership', 'train': ['i1'], 'validation': [], 'test': ['evaluation']}
    if task == 'forecasting':
        profile['release']['unit'] = 'series'
        protocol['split'] = {'kind': 'temporal', 'frequency': 'D',
                             'cases': [{'item_id': 'evaluation', 'series_id': 's1', 'partition': 'test',
                                        'origin': 5, 'fit_end': 5, 'target_end': 7, 'horizon': 2}]}
    if task == 'anomaly_detection':
        profile['release']['unit'] = 'event'
    plan = resolve_protocol(manifest)

    def record(model, value):
        return records.MetricRecord(f'out-{model}', 'run1', 'toy', model, 'evaluation', metric_name, value,
                                    coordinates={'repeat': 0},
                                    metadata={'plan_id': plan.plan_id, 'comparison_key': plan.comparison_key,
                                              'source': 'local_run', 'artifact_ref': 'out/metrics.json'}).to_dict()

    envelope = {'version': 'industrial_report_evidence@1', 'visibility': 'public_aggregates',
                'repeat_count': 1, 'items_by_dataset': {'toy': ['evaluation']},
                'metrics': [record('dummy', .75), record('other', .5), record('other', .5)],
                'provenance': {'code_revision': 'example', 'recorded_at': '2026-11-18T12:00:00Z'}}
    return manifest, plan, envelope


@pytest.mark.parametrize(('task', 'metric'), [('ts_classification', 'accuracy'), ('ts_regression', 'rmse'),
                                              ('forecasting', 'rmse'), ('anomaly_detection', 'vus_pr')])
def test_offline_report_saved_replay_and_coverage(task, metric, tmp_path):
    manifest, plan, envelope = _saved_report(task, metric)
    report = report_from_evidence(plan, envelope)
    assert report.successful == report.expected == 2
    assert report.ranked[0][0] == ('other' if metric == 'rmse' else 'dummy')
    assert pairwise_units(report, 'dummy', 'other')['matched'] == (('toy', 'evaluation', 0, .25),)
    manifest_path, evidence_path = tmp_path / 'manifest.json', tmp_path / 'evidence.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    evidence_path.write_text(json.dumps(envelope), encoding='utf-8')
    output = tmp_path / 'out'
    assert main([str(manifest_path), '--report-evidence', str(evidence_path), '--output-dir', str(output)]) == 0
    with (output / 'metrics.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    html = (output / 'index.html').read_text(encoding='utf-8')
    assert len(rows) == 2 and all(f'<td>{row["value"]}</td>' in html for row in rows)
    with pytest.raises(ProtocolError, match='public'):
        report_from_evidence(plan, {**envelope, 'visibility': 'private'})
    failed = {**envelope['metrics'][1], 'status': 'failed', 'value': None}
    partial = report_from_evidence(plan, {**envelope, 'metrics': [envelope['metrics'][0], failed]})
    assert partial.ranked == () and partial.rows[1].status == 'failed'
    assert pairwise_units(partial, 'dummy', 'other')['left_only'] == (('toy', 'evaluation', 0),)
    absent = report_from_evidence(plan, {**envelope, 'metrics': [envelope['metrics'][0]]})
    assert absent.rows[1].status == 'unavailable'
    mismatched = {**envelope['metrics'][0], 'metadata': {**envelope['metrics'][0]['metadata'], 'plan_id': 'other'}}
    with pytest.raises(ProtocolError, match='another plan'):
        report_from_evidence(plan, {**envelope, 'metrics': [mismatched]})
