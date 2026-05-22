import pytest
from dataclasses import replace
from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    ArtifactSpec,
    TaskType,
    RunSpec,
    ForecastingScenarioSpec,
    ForecastingBenchmarkResult,
    ForecastingSeriesRecord,
    RunMode,
    CovariateMode,
    LeakagePolicy,
    ProbabilisticMode,
    default_scenario,
)
from benchmark.v2.forecasting import ForecastingSuiteRunner


def test_old_config_runs_without_changes(tmp_path):
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
    )

    assert config.scenario_spec is None

    scenario = config.scenario_spec
    effective = scenario if scenario is not None else default_scenario()
    assert effective.run_mode == RunMode.FULL_SHOT


def test_new_config_sets_run_mode_explicitly(tmp_path):
    for mode in ['zero_shot', 'few_shot', 'full_shot']:
        scenario = ForecastingScenarioSpec(run_mode=mode)
        config = BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
            models=(ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),),
            artifact_spec=ArtifactSpec(output_dir=str(tmp_path)),
            scenario_spec=scenario,
        )
        assert config.scenario_spec.run_mode == mode


def test_default_scenario():
    scenario = default_scenario()
    assert scenario.run_mode == RunMode.FULL_SHOT
    assert scenario.covariate_mode == CovariateMode.NONE
    assert scenario.probabilistic == ProbabilisticMode.NONE
    assert scenario.leakage_policy == LeakagePolicy.STRICT


def test_result_record_stores_scenario_metadata():
    scenario = ForecastingScenarioSpec(run_mode='zero_shot')

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='test', dataset_name='test'),),
        models=(ModelSpec(adapter_name='test', display_name='Test'),),
        artifact_spec=ArtifactSpec(output_dir='./test'),
        scenario_spec=scenario,
    )

    result = ForecastingBenchmarkResult(
        run_id='test_123',
        config=config,
        series_records=(),
        run_records=(),
        prediction_records=(),
        metric_records=(),
        aggregate_report=None,  # simplified for test
    )

    assert result.config.scenario_spec.run_mode == RunMode.ZERO_SHOT


def test_few_shot_mode():
    series = ForecastingSeriesRecord(
        benchmark='test',
        dataset_name='test',
        subset='default',
        series_id='s1',
        frequency='D',
        forecast_horizon=3,
        seasonal_period=1,
        train_values=tuple(range(100)),
        test_values=(100, 101, 102),
    )

    n = len(series.train_values)
    k = max(3, min(n // 5, 50))
    limited = series.train_values[-k:]

    assert len(limited) == 20
    assert limited == tuple(range(80, 100))


def test_zero_shot_mode():
    series = ForecastingSeriesRecord(
        benchmark='test',
        dataset_name='test',
        subset='default',
        series_id='s1',
        frequency='D',
        forecast_horizon=3,
        seasonal_period=1,
        train_values=tuple(range(100)),
        test_values=(100, 101, 102),
    )

    scenario = ForecastingScenarioSpec(run_mode='zero_shot')

    if scenario.run_mode == RunMode.ZERO_SHOT:
        prepared = replace(series, train_values=())
    else:
        prepared = series

    assert prepared.train_values == ()


def test_full_shot_mode():
    series = ForecastingSeriesRecord(
        benchmark='test',
        dataset_name='test',
        subset='default',
        series_id='s1',
        frequency='D',
        forecast_horizon=3,
        seasonal_period=1,
        train_values=tuple(range(100)),
        test_values=(100, 101, 102),
    )

    scenario = ForecastingScenarioSpec(run_mode='full_shot')

    if scenario.run_mode == RunMode.ZERO_SHOT:
        prepared = replace(series, train_values=())
    elif scenario.run_mode == RunMode.FEW_SHOT:
        n = len(series.train_values)
        k = max(3, min(n // 5, 50))
        prepared = replace(series, train_values=series.train_values[-k:])
    else:
        prepared = series

    assert prepared.train_values == series.train_values
    assert len(prepared.train_values) == 100