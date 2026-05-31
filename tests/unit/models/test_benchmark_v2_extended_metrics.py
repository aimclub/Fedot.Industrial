import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import replace

from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    ArtifactSpec,
    TaskType,
    RunSpec,
    RunStatus,
    MetricKind,
    MetricRecord,
    ForecastResult,
    ForecastingSeriesRecord,
)
from benchmark.v2.forecasting import ForecastingSuiteRunner
from benchmark.v2.probabilistic_metrics import (
    pinball_loss,
    weighted_quantile_loss,
    crps,
    interval_coverage,
    calibration,
)


@pytest.fixture
def minimal_config(tmp_path):
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(ModelSpec(adapter_name='naive_last_value', display_name='Naive'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae', 'rmse'),
    )


@pytest.fixture
def dummy_series_record():
    return ForecastingSeriesRecord(
        benchmark='test',
        dataset_name='test',
        subset='default',
        series_id='s1',
        frequency='D',
        forecast_horizon=3,
        seasonal_period=7,
        train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        test_values=(6.0, 7.0, 8.0),
    )


@pytest.fixture
def dummy_forecast_result():
    """ForecastResult with quantiles and samples."""
    return ForecastResult(
        mean=[6.0, 7.0, 8.0],
        quantiles={
            0.1: [4.0, 5.0, 6.0],
            0.5: [6.0, 7.0, 8.0],
            0.9: [8.0, 9.0, 10.0],
        },
        intervals={
            0.8: ([4.0, 5.0, 6.0], [8.0, 9.0, 10.0]),
        },
        samples=[
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
        ],
    )


# ========== 1. Point metrics continue to work ==========

def test_point_metrics_compute_correctly():
    """Check that compute_forecasting_metric still works for point metrics."""
    from benchmark.v2.forecasting import compute_forecasting_metric

    y_true = np.array([6.0, 7.0, 8.0])
    y_pred = np.array([6.5, 7.5, 8.5])
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seasonal = 1

    mae = compute_forecasting_metric('mae', y_true, y_pred, y_train, seasonal)
    rmse = compute_forecasting_metric('rmse', y_true, y_pred, y_train, seasonal)
    smape = compute_forecasting_metric('smape', y_true, y_pred, y_train, seasonal)
    mase = compute_forecasting_metric('mase', y_true, y_pred, y_train, seasonal)
    owa = compute_forecasting_metric('owa', y_true, y_pred, y_train, seasonal)
    rmsse = compute_forecasting_metric('rmsse', y_true, y_pred, y_train, seasonal)
    wrmsse = compute_forecasting_metric('wrmsse', y_true, y_pred, y_train, seasonal)

    assert mae == pytest.approx(0.5)
    assert rmse == pytest.approx(0.5)
    assert 0 < smape < 100
    assert mase > 0
    assert owa > 0
    assert rmsse > 0
    assert wrmsse > 0


def test_probabilistic_metrics_not_applicable(minimal_config, dummy_series_record):
    """When forecast_result has no quantiles/samples -> NOT_APPLICABLE record."""
    runner = ForecastingSuiteRunner(minimal_config)
    runner.metric_records = []  # reset

    forecast_result_no_quantiles = ForecastResult(mean=[6.0, 7.0, 8.0])
    runner._record_probabilistic_metrics(
        dummy_series_record, "Naive", np.array([6.0, 7.0, 8.0]), forecast_result_no_quantiles
    )

    # Should create one NOT_APPLICABLE record
    applicable_records = [r for r in runner.metric_records if r.metric_name == 'probabilistic_not_applicable']
    assert len(applicable_records) == 1
    record = applicable_records[0]
    assert record.status == RunStatus.NOT_APPLICABLE
    assert record.kind == MetricKind.PROBABILISTIC
    assert record.metadata['reason'] == 'no quantiles or samples'


def test_probabilistic_metrics_are_computed_when_quantiles_present(
    minimal_config, dummy_series_record, dummy_forecast_result
):
    """When quantiles are present, probabilistic metrics should be computed."""
    runner = ForecastingSuiteRunner(minimal_config)
    runner.metric_records = []

    actual = np.array([6.0, 7.0, 8.0])
    runner._record_probabilistic_metrics(
        dummy_series_record, "ProbModel", actual, dummy_forecast_result
    )

    metric_names = [r.metric_name for r in runner.metric_records]
    assert any('pinball_q' in name for name in metric_names)
    assert 'wql' in metric_names
    assert 'crps' in metric_names
    assert any('coverage_' in name for name in metric_names)
    assert any('calibration_' in name for name in metric_names)

    for record in runner.metric_records:
        assert record.status == RunStatus.SUCCESS
        assert record.kind == MetricKind.PROBABILISTIC


def test_resource_metrics_are_recorded(minimal_config, dummy_series_record):
    runner = ForecastingSuiteRunner(minimal_config)
    runner.metric_records = []

    start_time = 10.0
    end_time = 12.5
    runner._record_resource_metrics(dummy_series_record, "ResourceModel", start_time, end_time)

    resource_records = [r for r in runner.metric_records if r.kind == MetricKind.RESOURCE]
    assert len(resource_records) >= 1
    metric_names = {r.metric_name for r in resource_records}
    assert 'wall_clock_sec' in metric_names
    assert 'memory_mb' in metric_names
    assert 'gpu_hours' in metric_names
    assert 'model_size_mb' in metric_names
    assert 'api_cost_usd' in metric_names

    wall_record = next(r for r in resource_records if r.metric_name == 'wall_clock_sec')
    assert wall_record.metric_value == 2.5
    assert wall_record.status == RunStatus.SUCCESS

    for name in ('memory_mb', 'gpu_hours', 'model_size_mb', 'api_cost_usd'):
        rec = next(r for r in resource_records if r.metric_name == name)
        assert rec.status == RunStatus.NOT_APPLICABLE
        assert rec.metadata['reason'] == 'not implemented yet'


def test_pinball_loss():
    actual = np.array([10, 20, 30])
    pred = np.array([12, 18, 32])
    loss = pinball_loss(actual, pred, quantile=0.5)
    assert loss == 1.0


def test_weighted_quantile_loss():
    actual = np.array([10, 20, 30])
    quantiles = {0.1: np.array([9, 18, 27]), 0.9: np.array([11, 22, 33])}
    wql = weighted_quantile_loss(actual, quantiles)
    assert wql > 0


def test_crps():
    actual = np.array([10, 20, 30])
    samples = np.array([
        [9, 19, 29],
        [11, 21, 31],
        [10, 20, 30],
    ])
    score = crps(actual, samples)
    assert 0 <= score <= 20


def test_interval_coverage():
    actual = np.array([5, 6, 7, 8])
    intervals = {0.8: (np.array([4, 5, 6, 7]), np.array([6, 7, 8, 9]))}
    cov = interval_coverage(actual, intervals, 0.8)
    assert cov == 1.0


def test_calibration():
    actual = np.array([1, 2, 3, 4])
    quantiles = {0.5: np.array([1, 3, 3, 4])}
    cal = calibration(actual, quantiles, 0.5)
    assert cal == 1.0


def test_record_probabilistic_metrics_handles_errors_gracefully(minimal_config, dummy_series_record):
    """If a probabilistic metric fails, it should record FAILED status."""
    runner = ForecastingSuiteRunner(minimal_config)
    runner.metric_records = []

    bad_result = ForecastResult(
        quantiles={0.5: [6.0, 7.0]},
    )
    actual = np.array([6.0, 7.0, 8.0])
    runner._record_probabilistic_metrics(dummy_series_record, "BadModel", actual, bad_result)

    failed = [r for r in runner.metric_records if r.status == RunStatus.FAILED]
    assert len(failed) >= 1
    assert failed[0].kind == MetricKind.PROBABILISTIC
    assert 'error' in failed[0].metadata
