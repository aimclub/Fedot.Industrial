from benchmark.v2 import (
    BenchmarkAggregateReport,
    BenchmarkRunRecord,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ForecastingBenchmarkResult,
    ModelSpec,
    OKHSSmoothingAcceptanceCriteria,
    RunSpec,
    RunStatus,
    TaskType,
    build_local_okhs_smoothing_suite_config,
    evaluate_okhs_smoothing_acceptance,
    summarize_okhs_smoothing_result,
)
from benchmark.v2.core import ArtifactSpec, ForecastingSeriesRecord


def test_build_local_okhs_smoothing_suite_config_has_expected_defaults() -> None:
    config = build_local_okhs_smoothing_suite_config(persist_on_run=False)

    assert config.task_type is TaskType.FORECASTING
    assert config.datasets[0].benchmark == 'm4'
    assert config.datasets[0].series_ids == ('D364', 'D377', 'D378')
    assert any(spec.adapter_name == 'okhs' for spec in config.models)


def test_summarize_okhs_smoothing_result_aggregates_collapse_and_correction() -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='m4', dataset_name='m4_daily_okhs_smoothing', subset='daily'),),
        models=(ModelSpec(adapter_name='okhs', display_name='OKHS DMD'),),
        artifact_spec=ArtifactSpec(output_dir='benchmark/results/test', persist_on_run=False),
        run_spec=RunSpec(run_name='quality_test'),
    )
    result = ForecastingBenchmarkResult(
        run_id='quality_test_123',
        config=config,
        series_records=(
            ForecastingSeriesRecord(
                benchmark='m4',
                dataset_name='m4_daily_okhs_smoothing',
                subset='daily',
                series_id='D364',
                frequency='daily',
                forecast_horizon=14,
                seasonal_period=1,
                train_values=(1.0, 2.0, 3.0),
                test_values=(4.0, 5.0),
            ),
        ),
        run_records=(
            BenchmarkRunRecord(
                run_id='quality_test_123',
                benchmark='m4',
                dataset_name='m4_daily_okhs_smoothing',
                subset='daily',
                series_id='D364',
                model_name='OKHS DMD',
                status=RunStatus.SUCCESS,
                metrics_summary={'mae': 1.2, 'smape': 8.0, 'mase': 0.9},
                metadata={
                    'fdmd_prediction_diagnostics': {
                        'anti_smoothing_diagnostics': {
                            'collapse_detected': True,
                            'correction_applied': True,
                            'collapse_resolved': True,
                            'envelope_ratio_before': 0.2,
                            'envelope_ratio_after': 0.6,
                            'forecast_amplitude_before': 0.03,
                            'forecast_amplitude_after': 0.12,
                        }
                    }
                },
            ),
        ),
        prediction_records=(),
        metric_records=(),
        aggregate_report=BenchmarkAggregateReport(
            run_id='quality_test_123',
            task_type=TaskType.FORECASTING,
            primary_metric='mae',
            leaderboard_rows=(),
            status_counts={'success': 1},
        ),
    )

    summary = summarize_okhs_smoothing_result(result)

    assert summary.series_count == 1
    assert summary.success_count == 1
    assert summary.collapse_count == 1
    assert summary.corrected_count == 1
    assert summary.resolved_count == 1
    assert summary.collapse_rate == 1.0
    assert summary.corrected_rate == 1.0
    assert summary.resolved_rate == 1.0
    assert summary.mean_mae == 1.2
    assert summary.mean_envelope_ratio_before == 0.2
    assert summary.mean_envelope_ratio_after == 0.6
    assert summary.mean_amplitude_gain == 0.09


def test_evaluate_okhs_smoothing_acceptance_reports_failure_reasons() -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='m4', dataset_name='m4_daily_okhs_smoothing', subset='daily'),),
        models=(ModelSpec(adapter_name='okhs', display_name='OKHS DMD'),),
        artifact_spec=ArtifactSpec(output_dir='benchmark/results/test', persist_on_run=False),
        run_spec=RunSpec(run_name='quality_test'),
    )
    result = ForecastingBenchmarkResult(
        run_id='quality_test_456',
        config=config,
        series_records=(),
        run_records=(
            BenchmarkRunRecord(
                run_id='quality_test_456',
                benchmark='m4',
                dataset_name='m4_daily_okhs_smoothing',
                subset='daily',
                series_id='D377',
                model_name='OKHS DMD',
                status=RunStatus.SUCCESS,
                metrics_summary={'mae': 2.0, 'smape': 12.0, 'mase': 1.5},
                metadata={
                    'fdmd_prediction_diagnostics': {
                        'anti_smoothing_diagnostics': {
                            'collapse_detected': True,
                            'correction_applied': False,
                            'collapse_resolved': False,
                            'envelope_ratio_before': 0.15,
                            'envelope_ratio_after': 0.15,
                            'forecast_amplitude_before': 0.02,
                            'forecast_amplitude_after': 0.02,
                        }
                    }
                },
            ),
        ),
        prediction_records=(),
        metric_records=(),
        aggregate_report=BenchmarkAggregateReport(
            run_id='quality_test_456',
            task_type=TaskType.FORECASTING,
            primary_metric='mae',
            leaderboard_rows=(),
            status_counts={'success': 1},
        ),
    )

    summary = summarize_okhs_smoothing_result(result)
    report = evaluate_okhs_smoothing_acceptance(
        summary,
        OKHSSmoothingAcceptanceCriteria(
            max_collapse_rate=1.0,
            min_corrected_rate=1.0,
            min_resolved_rate=1.0,
            min_mean_amplitude_gain=0.01,
            max_mean_envelope_ratio_after=0.5,
        ),
    )

    assert report.passed is False
    assert report.reasons
    assert any('corrected_rate' in reason for reason in report.reasons)
    assert any('resolved_rate' in reason for reason in report.reasons)
