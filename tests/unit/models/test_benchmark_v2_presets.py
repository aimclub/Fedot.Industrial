from __future__ import annotations

from benchmark.v2 import (
    ModelSpec,
    TaskType,
    build_local_m4_suite_config,
    build_local_monash_suite_config,
    build_local_okhs_smoothing_suite_config,
    build_local_tser_suite_config,
    build_local_ucr_suite_config,
    run_local_benchmark_preset,
)


def test_build_local_forecasting_presets_have_expected_defaults() -> None:
    m4_config = build_local_m4_suite_config(persist_on_run=False, sample_size=1)
    monash_config = build_local_monash_suite_config(persist_on_run=False, sample_size=1)

    assert m4_config.task_type is TaskType.FORECASTING
    assert monash_config.task_type is TaskType.FORECASTING
    assert m4_config.datasets[0].benchmark == 'm4'
    assert monash_config.datasets[0].benchmark == 'monash'
    assert m4_config.datasets[0].adapter_options['use_local_files'] is True
    assert monash_config.datasets[0].adapter_options['use_local_files'] is True


def test_build_local_task_presets_have_expected_defaults() -> None:
    ucr_config = build_local_ucr_suite_config(persist_on_run=False)
    tser_config = build_local_tser_suite_config(persist_on_run=False)

    assert ucr_config.task_type is TaskType.TS_CLASSIFICATION
    assert tser_config.task_type is TaskType.TS_REGRESSION
    assert ucr_config.datasets[0].benchmark == 'ucr'
    assert tser_config.datasets[0].benchmark == 'local_tser'


def test_build_local_okhs_smoothing_preset_has_expected_defaults() -> None:
    config = build_local_okhs_smoothing_suite_config(persist_on_run=False)

    assert config.task_type is TaskType.FORECASTING
    assert config.datasets[0].dataset_name == 'm4_daily_okhs_smoothing'
    assert config.datasets[0].series_ids == ('D364', 'D377', 'D378')
    assert any(spec.display_name == 'OKHS DMD' for spec in config.models)


def test_run_local_benchmark_preset_smoke_forecasting() -> None:
    result = run_local_benchmark_preset(
        'm4',
        sample_size=1,
        persist_on_run=False,
        models=(ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),),
    )

    assert result.config.task_type is TaskType.FORECASTING
    assert any(record.status.value == 'success' for record in result.run_records)


def test_run_local_benchmark_preset_smoke_tsc() -> None:
    result = run_local_benchmark_preset(
        'ucr',
        dataset_name='Lightning7',
        persist_on_run=False,
        models=(ModelSpec(adapter_name='nearest_centroid', display_name='NearestCentroid'),),
    )

    assert result.config.task_type is TaskType.TS_CLASSIFICATION
    assert any(record.status.value == 'success' for record in result.run_records)


def test_run_local_benchmark_preset_smoke_tser() -> None:
    result = run_local_benchmark_preset(
        'tser',
        dataset_name='NaturalGasPricesSentiment',
        persist_on_run=False,
        models=(ModelSpec(adapter_name='linear_regressor', display_name='LinearRegressor'),),
    )

    assert result.config.task_type is TaskType.TS_REGRESSION
    assert any(record.status.value == 'success' for record in result.run_records)


def test_run_local_benchmark_preset_smoke_okhs_smoothing() -> None:
    result = run_local_benchmark_preset(
        'okhs_smoothing',
        persist_on_run=False,
        models=(ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),),
    )

    assert result.config.task_type is TaskType.FORECASTING
    assert result.config.datasets[0].series_ids == ('D364', 'D377', 'D378')
    assert any(record.status.value == 'success' for record in result.run_records)
