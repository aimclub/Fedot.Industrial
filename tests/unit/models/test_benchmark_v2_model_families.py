import pytest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    BenchmarkAggregateReport,
    DatasetSpec,
    ModelSpec,
    MetricRecord,
    ArtifactSpec,
    TaskType,
    RunSpec,
    RunStatus,
    ModelFamily,
    BenchmarkRunRecord,
    ForecastingBenchmarkResult,
    PredictionRecord,
)
from benchmark.v2.forecasting import build_model_adapter, build_leaderboard
from benchmark.v2.analytics import render_publication_pack
from benchmark.v2.presets import run_local_benchmark_preset


def test_each_model_in_preset_has_family():
    model_specs = [
        ModelSpec(adapter_name='naive_last_value',
                  display_name='NaiveLastValue'),
        ModelSpec(adapter_name='moving_average', display_name='MovingAverage'),
        ModelSpec(adapter_name='linear_trend', display_name='LinearTrend'),
        ModelSpec(adapter_name='mssa', display_name='mSSA'),
        ModelSpec(adapter_name='havok', display_name='HAVOK'),
        ModelSpec(adapter_name='okhs', display_name='OKHS'),
        ModelSpec(adapter_name='patch_tst_model', display_name='PatchTST'),
        ModelSpec(adapter_name='autogluon', display_name='AutoGluon'),
    ]

    for spec in model_specs:
        model = build_model_adapter(spec)
        assert hasattr(model, 'family')
        assert model.family is not None
        assert model.family in list(ModelFamily)


def test_model_family_values_match_enum():

    assert build_model_adapter(
        ModelSpec(adapter_name='naive_last_value', display_name='Test')).family == ModelFamily.CLASSICAL_BASELINE
    assert build_model_adapter(
        ModelSpec(adapter_name='moving_average', display_name='Test')).family == ModelFamily.CLASSICAL_BASELINE
    assert build_model_adapter(
        ModelSpec(adapter_name='linear_trend', display_name='Test')).family == ModelFamily.CLASSICAL_BASELINE
    assert build_model_adapter(
        ModelSpec(adapter_name='classical_dmd', display_name='Test')).family == ModelFamily.CLASSICAL_BASELINE

    assert build_model_adapter(
        ModelSpec(adapter_name='mssa', display_name='Test')).family == ModelFamily.INTERNAL_INDUSTRIAL
    assert build_model_adapter(
        ModelSpec(adapter_name='havok', display_name='Test')).family == ModelFamily.INTERNAL_INDUSTRIAL
    assert build_model_adapter(
        ModelSpec(adapter_name='okhs', display_name='Test')).family == ModelFamily.INTERNAL_INDUSTRIAL

    assert build_model_adapter(
        ModelSpec(adapter_name='patch_tst_model', display_name='Test')).family == ModelFamily.SUPERVISED_SOTA

    assert build_model_adapter(ModelSpec(
        adapter_name='autogluon', display_name='Test')).family == ModelFamily.AUTOML


def test_optional_external_model_has_family():

    model = build_model_adapter(
        ModelSpec(adapter_name='nbeats', display_name='NBeats'))
    assert model.family == ModelFamily.EXTERNAL

    model = build_model_adapter(
        ModelSpec(adapter_name='tft', display_name='TFT'))
    assert model.family == ModelFamily.EXTERNAL


def test_benchmark_run_record_stores_family():

    record = BenchmarkRunRecord(
        run_id='test_123',
        benchmark='m4',
        dataset_name='daily',
        subset='default',
        series_id='series_1',
        model_name='NaiveLastValue',
        status=RunStatus.SUCCESS,
        family='classical_baseline',
    )

    assert record.family == 'classical_baseline'
    assert hasattr(record, 'family')


def test_leaderboard_groups_results_by_family():

    run_records = [
        BenchmarkRunRecord(
            run_id='1', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s1', model_name='NaiveLastValue', status=RunStatus.SUCCESS,
            metrics_summary={'mae': 1.0}, family='classical_baseline',
        ),
        BenchmarkRunRecord(
            run_id='2', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s2', model_name='NaiveLastValue', status=RunStatus.SUCCESS,
            metrics_summary={'mae': 1.1}, family='classical_baseline',
        ),
        BenchmarkRunRecord(
            run_id='3', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s1', model_name='mSSA', status=RunStatus.SUCCESS,
            metrics_summary={'mae': 0.8}, family='internal_industrial',
        ),
        BenchmarkRunRecord(
            run_id='4', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s2', model_name='mSSA', status=RunStatus.SUCCESS,
            metrics_summary={'mae': 0.9}, family='internal_industrial',
        ),
    ]

    report = build_leaderboard(tuple(run_records), primary_metric='mae')

    for row in report.leaderboard_rows:
        assert 'family' in row
        assert row['family'] in ['classical_baseline', 'internal_industrial']


def test_skipped_model_retains_family():

    spec = ModelSpec(adapter_name='autogluon', display_name='AutoGluon')
    model = build_model_adapter(spec)

    assert model.family == ModelFamily.AUTOML

    record = BenchmarkRunRecord(
        run_id='test',
        benchmark='m4',
        dataset_name='daily',
        subset='default',
        series_id='s1',
        model_name='AutoGluon',
        status=RunStatus.NOT_AVAILABLE,
        family=model.family,
        message='autogluon not installed',
    )

    assert record.status == RunStatus.NOT_AVAILABLE
    assert record.family == 'automl'


def test_all_classical_baseline_models_have_correct_family():
    classical_models = [
        'naive_last_value', 'naive_mean', 'naive_drift',
        'moving_average', 'linear_trend', 'classical_dmd',
    ]

    for adapter_name in classical_models:
        model = build_model_adapter(
            ModelSpec(adapter_name=adapter_name, display_name=adapter_name))
        assert model.family == ModelFamily.CLASSICAL_BASELINE, f"{adapter_name} has wrong family"


def test_all_internal_industrial_models_have_correct_family():
    internal_models = [
        'mssa', 'mssa_forecaster', 'havok', 'havok_forecaster',
        'okhs', 'okhs_fdmd_forecaster', 'ssa_forecaster',
        'lagged_forecaster', 'lagged_ridge_forecaster',
        'low_rank_lagged_ridge_forecaster', 'hybrid_ensemble_forecaster',
    ]

    for adapter_name in internal_models:
        model = build_model_adapter(
            ModelSpec(adapter_name=adapter_name, display_name=adapter_name))
        assert model.family == ModelFamily.INTERNAL_INDUSTRIAL, f"{adapter_name} has wrong family"


def test_all_supervised_sota_models_have_correct_family():
    sota_models = [
        'patch_tst_model', 'tst_model', 'tcn_model',
        'deepar_model', 'nbeats_model',
    ]

    for adapter_name in sota_models:
        model = build_model_adapter(
            ModelSpec(adapter_name=adapter_name, display_name=adapter_name))
        assert model.family == ModelFamily.SUPERVISED_SOTA, f"{adapter_name} has wrong family"


def test_build_leaderboard_includes_family():
    run_records = (
        BenchmarkRunRecord(run_id='1', benchmark='m4', dataset_name='daily', subset='default',
                           series_id='s1', model_name='Naive', status=RunStatus.SUCCESS,
                           metrics_summary={'mae': 1.0}, family='classical_baseline'),
        BenchmarkRunRecord(run_id='2', benchmark='m4', dataset_name='daily', subset='default',
                           series_id='s2', model_name='PatchTST', status=RunStatus.SUCCESS,
                           metrics_summary={'mae': 0.5}, family='supervised_sota'),
    )

    report = build_leaderboard(run_records, primary_metric='mae')

    for row in report.leaderboard_rows:
        assert 'family' in row
