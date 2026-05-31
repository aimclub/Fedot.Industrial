import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from benchmark.v2.core import (
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    ArtifactSpec,
    TaskType,
    RunSpec,
    RunStatus,
    ModelFamily,
    BenchmarkAggregateReport,
)
from benchmark.v2.forecasting import build_model_adapter, ForecastingSuiteRunner, ForecastingSeriesRecord, BenchmarkRunRecord
from benchmark.v2.api import run_forecasting_benchmark_suite


@pytest.fixture
def minimal_config(tmp_path):
    return BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(
            ModelSpec(adapter_name='naive_last_value', display_name='Naive'),
            ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
            ModelSpec(adapter_name='chronos', display_name='Chronos', optional=True),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )


@pytest.fixture
def mock_model_availability(monkeypatch):
    """Mock build_model_adapter to return a model with controlled availability."""
    def mock_build(spec):
        if spec.adapter_name == 'autogluon':
            model = MagicMock()
            model.name = 'AutoGluon'
            model.tags = ('external',)
            model.optional = True
            model.family = ModelFamily.AUTOML
            model.availability.return_value = (RunStatus.NOT_AVAILABLE, 'autogluon not installed')
            if hasattr(model, 'get_dependency_status'):
                model.get_dependency_status.return_value = {'autogluon': 'missing'}
            return model
        else:
            model = MagicMock()
            model.name = spec.display_name
            model.tags = ('baseline',)
            model.optional = False
            model.family = ModelFamily.CLASSICAL_BASELINE
            model.availability.return_value = (RunStatus.SUCCESS, 'ready')
            model.forecast.return_value = (np.array([1.0, 2.0, 3.0]), {})
            return model
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter', mock_build)


def test_external_models_absence_does_not_break_suite(tmp_path, monkeypatch):
    """Even when AutoGluon/Chronos are missing, the suite should complete successfully."""
    class DummyDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test',
                    subset='default',
                    series_id='s1',
                    frequency='D',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )
    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter', lambda spec: DummyDatasetAdapter())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(
            ModelSpec(adapter_name='naive_last_value', display_name='Naive'),
            ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    try:
        result = run_forecasting_benchmark_suite(config)
    except Exception as e:
        pytest.fail(f"Suite crashed due to external models absence: {e}")

    assert len(result.run_records) == 2

    autogluon_record = next(r for r in result.run_records if r.model_name == 'AutoGluon')
    assert autogluon_record.status == RunStatus.NOT_AVAILABLE
    assert autogluon_record.skip_reason == 'dependency_missing'
    assert autogluon_record.dependency_status is not None


def test_external_models_do_not_cause_crash_when_import_fails(monkeypatch):
    """Simulate that importing autogluon raises ImportError, but suite still works."""
    def fake_safe_import(module):
        if module == 'autogluon':
            return False
        return True
    monkeypatch.setattr('benchmark.v2.forecasting._safe_import', fake_safe_import)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),),
        artifact_spec=ArtifactSpec(output_dir='/tmp/unused', persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )
    class DummyDatasetAdapter:
        def load_series(self, spec):
            return ()
    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter', lambda spec: DummyDatasetAdapter())

    adapter = build_model_adapter(config.models[0])
    status, msg = adapter.availability()
    assert status == RunStatus.NOT_AVAILABLE
    assert 'autogluon' in msg


def test_aggregate_report_contains_status_counts_by_family(tmp_path, monkeypatch):
    """Check that the final report includes status_counts_by_family."""
    run_records = [
        BenchmarkRunRecord(
            run_id='1', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s1', model_name='Naive', family='classical_baseline',
            status=RunStatus.SUCCESS, metrics_summary={'mae': 1.0},
        ),
        BenchmarkRunRecord(
            run_id='2', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s2', model_name='MovingAverage', family='classical_baseline',
            status=RunStatus.FAILED, message='error',
        ),
        BenchmarkRunRecord(
            run_id='3', benchmark='m4', dataset_name='daily', subset='default',
            series_id='s3', model_name='AutoGluon', family='automl',
            status=RunStatus.NOT_AVAILABLE,
        ),
    ]

    from benchmark.v2.forecasting import build_leaderboard
    report = build_leaderboard(tuple(run_records), primary_metric='mae')

    assert hasattr(report, 'status_counts_by_family')
    counts = report.status_counts_by_family
    assert counts['classical_baseline']['success'] == 1
    assert counts['classical_baseline']['failed'] == 1
    assert counts['automl']['not_available'] == 1


def test_report_integration_with_external_models(tmp_path, monkeypatch):
    """End-to-end test: run a config with missing external models and verify report breakdown."""
    class DummyDatasetAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='test',
                    subset='default',
                    series_id='s1',
                    frequency='D',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0),
                    test_values=(5.0, 6.0, 7.0),
                ),
            )
    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter', lambda spec: DummyDatasetAdapter())

    def build_model_mock(spec):
        if spec.adapter_name == 'autogluon':
            model = MagicMock()
            model.name = 'AutoGluon'
            model.tags = ('external',)
            model.optional = True
            model.family = ModelFamily.AUTOML
            model.availability.return_value = (RunStatus.NOT_AVAILABLE, 'not installed')
            if hasattr(model, 'get_dependency_status'):
                model.get_dependency_status.return_value = {'autogluon': 'missing'}
            return model
        elif spec.adapter_name == 'naive_last_value':
            model = MagicMock()
            model.name = 'NaiveLastValue'
            model.tags = ('baseline',)
            model.optional = False
            model.family = ModelFamily.CLASSICAL_BASELINE
            model.availability.return_value = (RunStatus.SUCCESS, 'ready')
            model.forecast.return_value = (np.array([5.0, 6.0, 7.0]), {})
            return model
        else:
            model = MagicMock()
            model.name = spec.display_name
            model.family = ModelFamily.EXTERNAL
            model.availability.return_value = (RunStatus.NOT_AVAILABLE, 'unknown')
            return model
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter', build_model_mock)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )

    result = run_forecasting_benchmark_suite(config)

    report = result.aggregate_report
    assert hasattr(report, 'status_counts_by_family')
    counts_by_family = report.status_counts_by_family
    assert counts_by_family['classical_baseline']['success'] == 1
    assert counts_by_family['automl']['not_available'] == 1

    autogluon_record = next(r for r in result.run_records if r.model_name == 'AutoGluon')
    naive_record = next(r for r in result.run_records if r.model_name == 'NaiveLastValue')
    assert autogluon_record.status == RunStatus.NOT_AVAILABLE
    assert autogluon_record.family == 'automl'
    assert naive_record.status == RunStatus.SUCCESS
    assert naive_record.family == 'classical_baseline'


def test_budget_exceeded_status_and_budget_info(tmp_path, monkeypatch):
    """Simulate a model exceeding time budget and check budget_info is recorded."""
    class BudgetModel:
        name = 'BudgetModel'
        tags = ('test',)
        optional = False
        family = ModelFamily.SUPERVISED_SOTA
        def availability(self):
            return RunStatus.SUCCESS, 'ready'
        def forecast(self, series_record):
            from benchmark.v2.forecasting import ModelExecutionError
            raise ModelExecutionError(
                status=RunStatus.BUDGET_EXCEEDED,
                message='Time budget exceeded',
                budget_info={'time_limit_sec': 10, 'time_used_sec': 12},
            )
    def build_mock(spec):
        return BudgetModel()
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter', build_mock)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(ModelSpec(adapter_name='budget', display_name='BudgetModel'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )
    class DummyAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory', dataset_name='test', subset='default',
                    series_id='s1', frequency='D', forecast_horizon=3, seasonal_period=1,
                    train_values=(1,2,3,4), test_values=(5,6,7),
                ),
            )
    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter', lambda spec: DummyAdapter())

    result = run_forecasting_benchmark_suite(config)
    record = result.run_records[0]
    assert record.status == RunStatus.BUDGET_EXCEEDED
    assert record.budget_info is not None
    assert record.budget_info['time_limit_sec'] == 10
    assert record.budget_info['time_used_sec'] == 12


def test_dependency_status_populated_for_external_model(tmp_path, monkeypatch):
    """Check that dependency_status is correctly filled in BenchmarkRunRecord."""
    class CustomExternalModel:
        name = 'CustomExternal'
        tags = ('external',)
        optional = True
        family = ModelFamily.EXTERNAL
        def availability(self):
            return RunStatus.NOT_AVAILABLE, 'missing deps'


        def get_dependency_status(self):
            return {'dep1': 'missing', 'dep2': 'present'}

        def forecast(self, series_record):
            pass
    def build_mock(spec):
        return CustomExternalModel()
    monkeypatch.setattr('benchmark.v2.forecasting.build_model_adapter', build_mock)

    class DummyAdapter:
        def load_series(self, spec):
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory', dataset_name='test', subset='default',
                    series_id='s1', frequency='D', forecast_horizon=3, seasonal_period=1,
                    train_values=(1,2,3,4), test_values=(5,6,7),
                ),
            )
    monkeypatch.setattr('benchmark.v2.forecasting.build_dataset_adapter', lambda spec: DummyAdapter())

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(DatasetSpec(benchmark='in_memory', dataset_name='test'),),
        models=(ModelSpec(adapter_name='custom', display_name='CustomExternal'),),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(show_progress=False),
        metrics=('mae',),
    )
    result = run_forecasting_benchmark_suite(config)
    record = result.run_records[0]
    if hasattr(record, 'dependency_status'):
        assert record.dependency_status is not None
