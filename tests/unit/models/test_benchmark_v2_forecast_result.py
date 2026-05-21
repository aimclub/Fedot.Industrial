import numpy as np

from benchmark.v2.core import (
    BenchmarkSuiteConfig, DatasetSpec, ModelSpec, ArtifactSpec,
    TaskType, RunSpec, RunStatus, ForecastingSeriesRecord,
    ForecastResult,
)
from benchmark.v2.forecasting import run_forecasting_suite
from benchmark.v2.forecasting_result import (
    coerce_forecast_result,
)


def test_legacy_tuple_adapter_still_works(monkeypatch, tmp_path):
    """Test that legacy (prediction, metadata) tuple still works."""

    class LegacyModel:
        name = 'LegacyModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            return (5.0, 6.0, 7.0), {'contract': 'legacy'}

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: LegacyModel())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='legacy', display_name='LegacyModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_suite(config)

    assert result.run_records[0].status == RunStatus.SUCCESS
    assert len(result.prediction_records) == 3
    assert result.quantile_prediction_records == ()
    assert result.prediction_records[0].y_pred == 5.0


def test_forecast_result_mean_works(monkeypatch, tmp_path):
    """Test ForecastResult with mean only."""

    class MeanModel:
        name = 'MeanModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            return ForecastResult(mean=(5.0, 6.0, 7.0))

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: MeanModel())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='mean', display_name='MeanModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_suite(config)

    assert result.run_records[0].status == RunStatus.SUCCESS
    assert result.run_records[0].metadata['forecast_result_kind'] == 'mean'
    assert [r.y_pred for r in result.prediction_records] == [5.0, 6.0, 7.0]
    assert result.quantile_prediction_records == ()


def test_quantile_only_result_uses_median_and_records_quantiles(monkeypatch, tmp_path):
    """Test that quantile-only result uses median for point forecast and records quantiles."""

    class QuantileOnlyModel:
        name = 'QuantileOnlyModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            return ForecastResult(quantiles={
                0.1: (4.0, 5.0, 6.0),
                0.5: (5.0, 6.0, 7.0),
                0.9: (6.0, 7.0, 8.0),
            })

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: QuantileOnlyModel())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='quantile', display_name='QuantileOnlyModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_suite(config)

    assert result.run_records[0].status == RunStatus.SUCCESS
    assert result.run_records[0].metadata['point_forecast_fallback'] == 'quantile_0.5'
    assert len(result.prediction_records) == 3
    assert [r.y_pred for r in result.prediction_records] == [5.0, 6.0, 7.0]
    assert len(result.quantile_prediction_records) == 9

    q50_records = [r for r in result.quantile_prediction_records if r.quantile == 0.5]
    assert len(q50_records) == 3
    assert [r.y_pred for r in q50_records] == [5.0, 6.0, 7.0]


def test_samples_only_result_uses_sample_mean_fallback(monkeypatch, tmp_path):
    """Test that samples-only result uses sample mean for point forecast."""

    class SampleOnlyModel:
        name = 'SampleOnlyModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            return ForecastResult(samples=[
                (4.0, 5.0, 6.0),
                (6.0, 7.0, 8.0),
            ])

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: SampleOnlyModel())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='samples', display_name='SampleOnlyModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_suite(config)

    assert result.run_records[0].status == RunStatus.SUCCESS
    assert result.run_records[0].metadata['point_forecast_fallback'] == 'sample_mean'
    assert [r.y_pred for r in result.prediction_records] == [5.0, 6.0, 7.0]


def test_quantile_horizon_mismatch_fails_controlled(monkeypatch, tmp_path):
    """Test that quantile horizon mismatch causes FAILED run."""

    class MismatchedQuantileModel:
        name = 'MismatchedQuantileModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            # Wrong horizon: length 2 instead of 3
            return ForecastResult(quantiles={0.5: (5.0, 6.0)})

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: MismatchedQuantileModel())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='mismatch', display_name='MismatchedQuantileModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_suite(config)

    assert result.run_records[0].status == RunStatus.FAILED
    assert 'length' in result.run_records[0].message.lower()


def test_persistence_resume_round_trip_includes_quantile_records(tmp_path, monkeypatch):
    """Test that quantile records are persisted and restored on resume."""

    output_dir = tmp_path / 'benchmark_output'

    class QuantileModel:
        name = 'QuantileModel'
        tags = ('test',)
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            return ForecastResult(quantiles={
                0.1: (4.0, 5.0, 6.0),
                0.5: (5.0, 6.0, 7.0),
                0.9: (6.0, 7.0, 8.0),
            })

    class FakeDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test_dataset',
                    subset='default',
                    series_id='series_1',
                    frequency='daily',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )

    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter',
                        lambda spec, include_optional_external=False: FakeDatasetAdapter())
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter',
                        lambda spec, include_optional_external=False: QuantileModel())

    # First run with persistence
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='quantile', display_name='QuantileModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(output_dir), persist_on_run=True),
        run_spec=RunSpec(show_progress=False, resume_enabled=False),
        metrics=('mae',),
    )

    first_result = run_forecasting_suite(config)
    assert len(first_result.quantile_prediction_records) == 9

    # Second run with resume
    resume_config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test_dataset'),),
        models=(ModelSpec(adapter_name='quantile', display_name='QuantileModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(output_dir), persist_on_run=True),
        run_spec=RunSpec(
            show_progress=False,
            resume_enabled=True,
            resume_run_id=first_result.run_id,
        ),
        metrics=('mae',),
    )

    resumed_result = run_forecasting_suite(resume_config)
    assert len(resumed_result.quantile_prediction_records) == 9
    assert resumed_result.run_records[0].model_name == 'QuantileModel'


def test_coerce_forecast_result_preserves_forecast_result():
    """coerce_forecast_result should pass through ForecastResult unchanged."""
    original = ForecastResult(mean=(1.0, 2.0, 3.0), metadata={'test': 123})
    coerced = coerce_forecast_result(original)
    assert coerced is original


def test_coerce_legacy_tuple():
    """coerce_forecast_result should convert legacy tuple."""
    raw = ([1.0, 2.0, 3.0], {'source': 'legacy'})
    result = coerce_forecast_result(raw)
    assert result.mean is not None
    assert np.array_equal(result.mean, [1.0, 2.0, 3.0])
    assert result.metadata['source'] == 'legacy'


def test_coerce_bare_list():
    """coerce_forecast_result should convert bare list."""
    raw = [1.0, 2.0, 3.0]
    result = coerce_forecast_result(raw)
    assert result.mean is not None
    assert np.array_equal(result.mean, [1.0, 2.0, 3.0])


def test_describe_forecast_result_kind():
    """describe_forecast_result_kind should correctly describe contents."""
    result = ForecastResult(mean=(1.0, 2.0), quantiles={0.5: (1.0, 2.0)})
    from benchmark.v2.forecasting_result import describe_forecast_result_kind
    assert 'mean' in describe_forecast_result_kind(result)
    assert 'quantiles' in describe_forecast_result_kind(result)
