from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.industrial.api import compare_forecasting_models_on_series, run_forecasting_benchmark_suite
from benchmark.industrial.core import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ForecastingSeriesRecord,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
)
from benchmark.industrial.forecasting import (
    _resolve_stage_tuning_split_spec,
    build_dataset_adapter,
    build_model_adapter,
    run_forecasting_suite,
)


def _toy_records() -> list[dict]:
    base = np.linspace(1.0, 24.0, num=24)
    return [
        {
            'series_id': 'toy_1',
            'values': (base + 0.5 * np.sin(np.arange(24))).tolist(),
            'horizon': 4,
            'frequency': 'monthly',
            'seasonal_period': 4,
            'dataset_name': 'toy_dataset',
        },
        {
            'series_id': 'toy_2',
            'values': (base * 1.2 + np.cos(np.arange(24))).tolist(),
            'horizon': 4,
            'frequency': 'monthly',
            'seasonal_period': 4,
            'dataset_name': 'toy_dataset',
        },
    ]


def _switching_records() -> list[dict]:
    time = np.arange(72, dtype=float)
    series = np.sin(2.0 * np.pi * time / 10.0)
    series[20:28] += 3.0
    series[40:52] -= 2.2
    series[52:] += np.linspace(0.0, 1.8, num=len(series[52:]))
    return [
        {
            'series_id': 'switch_1',
            'values': series.tolist(),
            'horizon': 6,
            'frequency': 'daily',
            'seasonal_period': 7,
            'dataset_name': 'switch_dataset',
        },
    ]


def test_build_model_adapter_supports_lagged_and_ssa_forecasters() -> None:
    lagged_model = build_model_adapter(
        ModelSpec(adapter_name='lagged_forecaster',
                  display_name='lagged_forecaster')
    )
    lagged_model_with_channel = build_model_adapter(
        ModelSpec(
            adapter_name='lagged_forecaster',
            display_name='lagged_forecaster_ridge',
            params={'channel_model': 'ridge'},
        )
    )
    ssa_model = build_model_adapter(
        ModelSpec(adapter_name='ssa_forecaster', display_name='ssa_forecaster')
    )

    assert lagged_model.name == 'lagged_forecaster'
    assert lagged_model_with_channel.channel_model == 'ridge'
    assert ssa_model.name == 'ssa_forecaster'


def test_lagged_forecaster_adapter_reports_unsupported_channel_model() -> None:
    lagged_model = build_model_adapter(
        ModelSpec(
            adapter_name='lagged_forecaster',
            display_name='lagged_forecaster_lasso',
            params={'channel_model': 'lasso'},
        )
    )

    status, message = lagged_model.availability()

    assert status is RunStatus.NOT_AVAILABLE
    assert "channel_model='ridge'" in message


def test_resolve_stage_tuning_split_spec_supports_time_series_cv_fields() -> None:
    split_spec = _resolve_stage_tuning_split_spec(
        {
            'split_spec': {
                'kind': 'time_series_split',
                'validation_horizon': 6,
                'n_splits': 4,
                'gap': 2,
                'max_train_size': 48,
            }
        }
    )

    assert split_spec is not None
    assert split_spec.kind.value == 'time_series_split'
    assert split_spec.n_splits == 4
    assert split_spec.gap == 2
    assert split_spec.max_train_size == 48


def test_run_forecasting_suite_adds_post_fit_stage_tuning_comparison(monkeypatch) -> None:
    captured_specs = []
    captured_tuning_kwargs = {}

    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='debug_suite',
                    subset='monthly',
                    series_id='series_1',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def __init__(self, scale: float = 1.0):
            self.scale = float(scale)

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            target = np.asarray(series_record.test_values, dtype=float)
            if self.scale > 1.0:
                return target.copy(), {'source': 'tuned'}
            return np.zeros_like(target), {'source': 'baseline'}

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())

    def _fake_build_model_adapter(spec):
        captured_specs.append(spec)
        return FakeForecastModel(scale=float(spec.params.get('scale', 1.0)))

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter', _fake_build_model_adapter)
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_forecasting_stage_tuning_on_series',
        lambda *args, **kwargs: captured_tuning_kwargs.update(kwargs) or type(
            'FakeStageTuningRuntimeResult',
            (),
            {
                'metadata': {'improved': True, 'baseline_score': 10.0, 'best_score': 0.0},
                'to_dict': lambda self: {
                    'sequential_result': {'best_parameters': {'scale': 2.0}},
                    'baseline_evaluation': {'metric': {'metric_value': 10.0}},
                    'best_evaluation': {'metric': {'metric_value': 0.0}},
                },
            },
        )(),
        raising=False,
    )

    result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='debug_suite', subset='monthly'),),
            models=(ModelSpec(adapter_name='fake_model',
                    display_name='FakeForecastModel'),),
            artifact_spec=ArtifactSpec(
                output_dir='unused', persist_on_run=False),
            run_spec=RunSpec(run_name='debug_suite_runner',
                             show_progress=False),
            metrics=('mae', 'rmse'),
        )
    )

    record = result.run_records[0]
    comparison = record.metadata['stage_tuning_comparison']

    assert comparison['baseline_metrics']['mae'] > comparison['tuned_metrics']['mae']
    assert comparison['improved_metrics']['mae'] is True
    assert comparison['best_parameters']['scale'] == 2.0
    assert any(metric.metric_name ==
               'mae_tuned' for metric in result.metric_records)
    assert captured_specs
    assert captured_specs[0].params['progress_policy']['enabled'] is False
    assert captured_tuning_kwargs['progress_policy'].enabled is False


def test_run_forecasting_suite_compact_verbosity_trims_stage_tuning_payload(monkeypatch) -> None:
    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='compact_suite',
                    subset='monthly',
                    series_id='series_1',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def __init__(self, scale: float = 1.0):
            self.scale = float(scale)

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            target = np.asarray(series_record.test_values, dtype=float)
            if self.scale > 1.0:
                return target.copy(), {'source': 'tuned', 'debug_blob': {'weights': [1, 2, 3]}}
            return np.zeros_like(target), {'source': 'baseline'}

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter',
        lambda spec: FakeForecastModel(
            scale=float(spec.params.get('scale', 1.0))),
    )
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_forecasting_stage_tuning_on_series',
        lambda *args, **kwargs: type(
            'FakeStageTuningRuntimeResult',
            (),
            {
                'metadata': {'improved': True, 'baseline_score': 10.0, 'best_score': 0.0},
                'to_dict': lambda self: {
                    'sequential_result': {
                        'best_parameters': {'scale': 2.0},
                        'stage_history': [{'stage': 'forecast_head', 'evaluations': [{'score': 1.0}]}],
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                    'baseline_evaluation': {
                        'metric': {'metric_value': 10.0},
                        'split_metadata': {'folds': [{'fold_index': 0}]},
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                    'best_evaluation': {
                        'metric': {'metric_value': 0.0},
                        'split_metadata': {'folds': [{'fold_index': 0}]},
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                },
            },
        )(),
        raising=False,
    )

    result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='compact_suite', subset='monthly'),),
            models=(ModelSpec(
                adapter_name='fake_model',
                display_name='FakeForecastModel',
                params={'stage_tuning_runtime': {}},
            ),),
            artifact_spec=ArtifactSpec(
                output_dir='unused', persist_on_run=False),
            run_spec=RunSpec(run_name='compact_suite_runner',
                             show_progress=False, verbosity='compact'),
            metrics=('mae', 'rmse'),
        )
    )

    record = result.run_records[0]
    report = record.metadata['stage_tuning_report']
    comparison = record.metadata['stage_tuning_comparison']

    assert 'evaluations' not in report['sequential_result']['stage_history'][0]
    assert 'split_metadata' not in report['baseline_evaluation'] or 'folds' not in report['baseline_evaluation'][
        'split_metadata']
    assert 'split_metadata' not in report['best_evaluation'] or 'folds' not in report['best_evaluation'][
        'split_metadata']
    assert 'progress_policy' not in report['sequential_result'].get(
        'metadata', {})
    assert 'progress_policy' not in record.metadata['stage_tuning_runtime']
    assert 'tuned_forecast' not in comparison
    assert 'tuned_metadata' not in comparison


def test_run_forecasting_suite_persists_incremental_item_artifacts(tmp_path: Path, monkeypatch) -> None:
    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='incremental_suite',
                    subset='monthly',
                    series_id='series_success',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='incremental_suite',
                    subset='monthly',
                    series_id='series_failed',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(2.0, 3.0, 4.0, 5.0, 6.0),
                    test_values=(7.0, 8.0, 9.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            if series_record.series_id == 'series_failed':
                raise RuntimeError('forced benchmark failure')
            target = np.asarray(series_record.test_values, dtype=float)
            return target.copy(), {'source': 'success'}

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter',
        lambda spec: FakeForecastModel(),
    )

    result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='incremental_suite', subset='monthly'),),
            models=(ModelSpec(adapter_name='fake_model',
                    display_name='FakeForecastModel'),),
            artifact_spec=ArtifactSpec(
                output_dir=str(tmp_path), persist_on_run=True),
            run_spec=RunSpec(
                run_name='incremental_suite_runner', show_progress=False),
            metrics=('mae', 'rmse'),
        )
    )

    progress_dir = tmp_path / result.run_id / 'progress'
    items_dir = progress_dir / 'items'
    progress_index = json.loads(
        (progress_dir / 'run_progress.json').read_text(encoding='utf-8'))
    item_paths = sorted(items_dir.glob('*.json'))

    assert (progress_dir / 'run_context.json').exists()
    assert (progress_dir / 'series_records.json').exists()
    assert (progress_dir / 'run_progress.json').exists()
    assert len(item_paths) == 2
    assert progress_index['completed_items'] == 2
    assert progress_index['status_counts']['success'] == 1
    assert progress_index['status_counts']['failed'] == 1

    success_payload = json.loads(item_paths[0].read_text(encoding='utf-8'))
    failure_payload = json.loads(item_paths[1].read_text(encoding='utf-8'))
    payloads = {
        success_payload['run_record']['series_id']: success_payload,
        failure_payload['run_record']['series_id']: failure_payload,
    }

    assert payloads['series_success']['metric_records']
    assert payloads['series_success']['prediction_records']
    assert payloads['series_failed']['metric_records'] == []
    assert payloads['series_failed']['prediction_records'] == []
    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'run_context.json' in artifact_names
    assert 'series_records.json' in artifact_names
    assert 'run_progress.json' in artifact_names


def test_run_forecasting_suite_resume_mode_skips_previously_persisted_items(tmp_path: Path, monkeypatch) -> None:
    forecast_calls = {'count': 0, 'fail_on_forecast': False}

    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='resume_suite',
                    subset='monthly',
                    series_id='series_1',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            del series_record
            if forecast_calls['fail_on_forecast']:
                raise RuntimeError(
                    'resume mode should not recompute persisted items')
            forecast_calls['count'] += 1
            return np.asarray([6.0, 7.0, 8.0], dtype=float), {'source': 'fresh'}

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter',
        lambda spec: FakeForecastModel(),
    )

    initial_result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='resume_suite', subset='monthly'),),
            models=(ModelSpec(adapter_name='fake_model',
                    display_name='FakeForecastModel'),),
            artifact_spec=ArtifactSpec(
                output_dir=str(tmp_path), persist_on_run=True),
            run_spec=RunSpec(run_name='resume_suite_runner',
                             show_progress=False),
            metrics=('mae', 'rmse'),
        )
    )

    assert forecast_calls['count'] == 1

    forecast_calls['fail_on_forecast'] = True
    resumed_result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='resume_suite', subset='monthly'),),
            models=(ModelSpec(adapter_name='fake_model',
                    display_name='FakeForecastModel'),),
            artifact_spec=ArtifactSpec(
                output_dir=str(tmp_path), persist_on_run=True),
            run_spec=RunSpec(
                run_name='resume_suite_runner',
                show_progress=False,
                resume_enabled=True,
                resume_run_id=initial_result.run_id,
            ),
            metrics=('mae', 'rmse'),
        )
    )

    progress_dir = tmp_path / initial_result.run_id / 'progress'
    progress_index = json.loads(
        (progress_dir / 'run_progress.json').read_text(encoding='utf-8'))

    assert resumed_result.run_id == initial_result.run_id
    assert forecast_calls['count'] == 1
    assert len(resumed_result.run_records) == 1
    assert len(resumed_result.metric_records) == len(
        initial_result.metric_records)
    assert len(resumed_result.prediction_records) == len(
        initial_result.prediction_records)
    assert progress_index['completed_items'] == 1


def test_forecasting_resume_from_empty_progress_preserves_fresh_metric_records(
        tmp_path: Path,
        monkeypatch,
) -> None:
    forecast_calls = {'count': 0}
    run_id = 'empty_progress_resume_001'
    progress_dir = tmp_path / run_id / 'progress'
    progress_dir.mkdir(parents=True)
    (progress_dir / 'run_progress.json').write_text(
        json.dumps({'run_id': run_id, 'completed_items': 0,
                   'status_counts': {}, 'item_artifacts': {}}),
        encoding='utf-8',
    )

    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='empty_progress_resume',
                    subset='monthly',
                    series_id='series_1',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            forecast_calls['count'] += 1
            return np.asarray(series_record.test_values, dtype=float), {'source': 'fresh_after_empty_resume'}

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter', lambda spec: FakeForecastModel())

    result = run_forecasting_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='empty_progress_resume', subset='monthly'),),
            models=(ModelSpec(adapter_name='fake_model',
                    display_name='FakeForecastModel'),),
            artifact_spec=ArtifactSpec(
                output_dir=str(tmp_path), persist_on_run=True),
            run_spec=RunSpec(
                run_name='empty_progress_resume',
                show_progress=False,
                resume_enabled=True,
                resume_run_id=run_id,
            ),
            metrics=('mae', 'rmse'),
        )
    )

    item_path = next((progress_dir / 'items').glob('*.json'))
    item_payload = json.loads(item_path.read_text(encoding='utf-8'))

    assert forecast_calls['count'] == 1
    assert len(result.metric_records) > 0
    assert len(result.prediction_records) == 3
    assert len(item_payload['metric_records']) == len(result.metric_records)
    assert len(item_payload['prediction_records']
               ) == len(result.prediction_records)


def test_publication_pack_compact_verbosity_trims_stage_tuning_artifacts(tmp_path: Path, monkeypatch) -> None:
    class FakeDatasetAdapter:
        def load_series(self, spec):
            del spec
            return (
                ForecastingSeriesRecord(
                    benchmark='in_memory',
                    dataset_name='compact_publication_suite',
                    subset='monthly',
                    series_id='series_1',
                    frequency='monthly',
                    forecast_horizon=3,
                    seasonal_period=1,
                    train_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                    test_values=(6.0, 7.0, 8.0),
                ),
            )

    class FakeForecastModel:
        name = 'FakeForecastModel'
        tags = ('baseline', 'forecasting')
        optional = False

        def __init__(self, scale: float = 1.0):
            self.scale = float(scale)

        def availability(self):
            return RunStatus.SUCCESS, 'ready'

        def forecast(self, series_record):
            target = np.asarray(series_record.test_values, dtype=float)
            if self.scale > 1.0:
                return target.copy(), {
                    'source': 'tuned',
                    'benchmark_runtime_context': {'debug': True},
                    'progress_policy': {'enabled': False},
                }
            return np.zeros_like(target), {
                'source': 'baseline',
                'benchmark_runtime_context': {'debug': True},
                'progress_policy': {'enabled': False},
            }

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_dataset_adapter', lambda spec: FakeDatasetAdapter())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_model_adapter',
        lambda spec: FakeForecastModel(
            scale=float(spec.params.get('scale', 1.0))),
    )
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_forecasting_stage_tuning_on_series',
        lambda *args, **kwargs: type(
            'FakeStageTuningRuntimeResult',
            (),
            {
                'metadata': {'improved': True, 'baseline_score': 10.0, 'best_score': 0.0},
                'to_dict': lambda self: {
                    'sequential_result': {
                        'best_parameters': {'scale': 2.0},
                        'stage_history': [{'stage': 'forecast_head', 'evaluations': [{'score': 1.0}]}],
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                    'baseline_evaluation': {
                        'metric': {'metric_value': 10.0},
                        'split_metadata': {'folds': [{'fold_index': 0}]},
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                    'best_evaluation': {
                        'metric': {'metric_value': 0.0},
                        'split_metadata': {'folds': [{'fold_index': 0}]},
                        'metadata': {'progress_policy': {'enabled': False}},
                    },
                },
            },
        )(),
        raising=False,
    )

    result = run_forecasting_benchmark_suite(
        BenchmarkSuiteConfig(
            task_type=TaskType.FORECASTING,
            datasets=(DatasetSpec(benchmark='in_memory',
                      dataset_name='compact_publication_suite', subset='monthly'),),
            models=(ModelSpec(
                adapter_name='fake_model',
                display_name='FakeForecastModel',
                params={'stage_tuning_runtime': {}},
            ),),
            artifact_spec=ArtifactSpec(
                output_dir=str(tmp_path), persist_on_run=True),
            run_spec=RunSpec(
                run_name='compact_publication_suite',
                show_progress=False,
                primary_metric='mae',
                verbosity='compact',
            ),
            metrics=('mae', 'rmse'),
        )
    )

    stage_tuning_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'series_1_forecasting_stage_tuning.json'
    )
    stage_payload = json.loads(stage_tuning_path.read_text(encoding='utf-8'))
    report = stage_payload['FakeForecastModel']

    assert 'evaluations' not in report['sequential_result']['stage_history'][0]
    assert 'split_metadata' not in report['baseline_evaluation'] or 'folds' not in report['baseline_evaluation'][
        'split_metadata']
    assert 'progress_policy' not in report['sequential_result'].get(
        'metadata', {})

    metadata_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'run_metadata.json'
    )
    metadata_payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert metadata_payload['verbosity_policy']['level'] == 'compact'


def test_build_model_adapter_supports_new_composite_forecasters() -> None:
    lagged_ridge_model = build_model_adapter(
        ModelSpec(adapter_name='lagged_ridge_forecaster',
                  display_name='lagged_ridge_forecaster')
    )
    low_rank_model = build_model_adapter(
        ModelSpec(adapter_name='low_rank_lagged_ridge_forecaster',
                  display_name='low_rank_lagged_ridge_forecaster')
    )
    hybrid_model = build_model_adapter(
        ModelSpec(adapter_name='hybrid_ensemble_forecaster',
                  display_name='hybrid_ensemble_forecaster')
    )
    okhs_fdmd_model = build_model_adapter(
        ModelSpec(adapter_name='okhs_fdmd_forecaster',
                  display_name='okhs_fdmd_forecaster')
    )

    assert lagged_ridge_model.name == 'lagged_ridge_forecaster'
    assert low_rank_model.name == 'low_rank_lagged_ridge_forecaster'
    assert hybrid_model.name == 'hybrid_ensemble_forecaster'
    assert okhs_fdmd_model.name == 'okhs_fdmd_forecaster'


def test_build_model_adapter_supports_canonical_mssa_and_havok_names() -> None:
    mssa_model = build_model_adapter(
        ModelSpec(adapter_name='mssa_forecaster',
                  display_name='mssa_forecaster')
    )
    havok_model = build_model_adapter(
        ModelSpec(adapter_name='havok_forecaster',
                  display_name='havok_forecaster')
    )

    assert mssa_model.name == 'mssa_forecaster'
    assert havok_model.name == 'havok_forecaster'


def test_build_model_adapter_supports_short_forecasting_aliases() -> None:
    mssa_model = build_model_adapter(
        ModelSpec(adapter_name='mssa', display_name='mssa')
    )
    havok_model = build_model_adapter(
        ModelSpec(adapter_name='havok', display_name='havok')
    )

    assert mssa_model.name == 'mssa'
    assert havok_model.name == 'havok'


def test_build_model_adapter_supports_mssa_head_runtime_parameters() -> None:
    mssa_model = build_model_adapter(
        ModelSpec(
            adapter_name='mssa_forecaster',
            display_name='mssa_tuned',
            params={'head_policy': 'mlp',
                    'head_hidden_dim': 96, 'head_epochs': 40},
        )
    )

    assert mssa_model.head_policy == 'mlp'
    assert mssa_model.head_hidden_dim == 96
    assert mssa_model.head_epochs == 40


def test_build_model_adapter_supports_havok_mlp_head_runtime_parameters() -> None:
    havok_model = build_model_adapter(
        ModelSpec(
            adapter_name='havok_forecaster',
            display_name='havok_tuned',
            params={
                'head_policy': 'mlp',
                'head_activation': 'gelu',
                'head_depth': 6,
                'head_base_hidden_dim': 256,
            },
        )
    )

    assert havok_model.head_policy == 'mlp'
    assert havok_model.head_activation == 'gelu'
    assert havok_model.head_depth == 6
    assert havok_model.head_base_hidden_dim == 256


@pytest.mark.parametrize(
    ('adapter_name', 'display_name'),
    (
        ('patch_tst_model', 'PatchTST'),
        ('tst_model', 'TST'),
        ('tcn_model', 'TCN'),
        ('deepar_model', 'DeepAR'),
        ('nbeats_model', 'NBEATS'),
    ),
)
def test_build_model_adapter_supports_native_neural_forecasting_heads(
        adapter_name: str,
        display_name: str,
) -> None:
    neural_model = build_model_adapter(
        ModelSpec(adapter_name=adapter_name, display_name=display_name)
    )

    assert neural_model.name == display_name
    assert neural_model.neural_model_name == adapter_name


def test_m4_adapter_parses_frame_and_samples() -> None:
    rows = []
    for series_id in ('M4_monthly_1', 'M4_monthly_2'):
        for step in range(20):
            rows.append(
                {
                    'unique_id': series_id,
                    'ds': step,
                    'y': float(step),
                    'horizon': 4,
                    'frequency': 'Monthly',
                    'seasonal_period': 12,
                }
            )
    frame = pd.DataFrame(rows)
    spec = DatasetSpec(
        benchmark='m4',
        dataset_name='toy_m4',
        subset='monthly',
        sample_size=1,
        random_seed=3,
        adapter_options={'loader': lambda _: frame},
    )

    adapter = build_dataset_adapter(spec)
    records = adapter.load_series(spec)

    assert len(records) == 1
    assert records[0].forecast_horizon == 4
    assert records[0].seasonal_period == 12
    assert len(records[0].train_values) == 16
    assert len(records[0].test_values) == 4


def test_m4_adapter_reads_local_repo_files() -> None:
    spec = DatasetSpec(
        benchmark='m4',
        dataset_name='m4_daily_local',
        subset='daily',
        sample_size=2,
        random_seed=1,
        adapter_options={'use_local_files': True},
    )

    adapter = build_dataset_adapter(spec)
    records = adapter.load_series(spec)

    assert len(records) == 2
    assert all(record.forecast_horizon == 14 for record in records)
    assert all(record.metadata['split_provenance'] ==
               'local_m4_train_test_csv' for record in records)


def test_monash_adapter_reads_local_repo_files() -> None:
    spec = DatasetSpec(
        benchmark='monash',
        dataset_name='Bitcoin',
        subset='daily',
        sample_size=2,
        random_seed=2,
        adapter_options={'use_local_files': True},
    )

    adapter = build_dataset_adapter(spec)
    records = adapter.load_series(spec)

    assert len(records) == 2
    assert all(record.forecast_horizon == 30 for record in records)
    assert all(record.metadata['split_provenance'] ==
               'local_monash_csv' for record in records)
    assert all(record.metadata['source_file'] ==
               'MonashBitcoin_30.csv' for record in records)


def test_m4_adapter_reads_local_repo_files_from_foreign_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = DatasetSpec(
        benchmark='m4',
        dataset_name='m4_daily_local',
        subset='daily',
        sample_size=1,
        random_seed=3,
        adapter_options={'use_local_files': True},
    )

    adapter = build_dataset_adapter(spec)
    records = adapter.load_series(spec)

    assert len(records) == 1
    assert records[0].metadata['split_provenance'] == 'local_m4_train_test_csv'


def test_forecasting_suite_runs_and_writes_publication_pack(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(adapter_name='naive_last_value',
                      display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average',
                      display_name='MovingAverage', params={'window_size': 3}),
            ModelSpec(adapter_name='mssa', display_name='mSSA',
                      params={'window_size': 6, 'rank': 2}),
            ModelSpec(
                adapter_name='okhs',
                display_name='OKHS Direct',
                params={
                    'method': 'direct',
                    'window_size': 8,
                    'n_modes': 2,
                    'q': 0.9,
                },
            ),
            ModelSpec(adapter_name='autogluon',
                      display_name='AutoGluon', optional=True),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert result.aggregate_report.primary_metric == 'mae'
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(
        record.model_name == 'OKHS Direct' and record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name ==
               'mSSA' and record.status is RunStatus.SUCCESS for record in result.run_records)
    okhs_record = next(
        record for record in result.run_records if record.model_name == 'OKHS Direct')
    assert okhs_record.metadata['method'] == 'direct'
    assert 'regime_diagnostics' in okhs_record.metadata
    assert 'routing_recommendation' in okhs_record.metadata
    assert any(record.horizon_index is None for record in result.metric_records)
    assert any(
        record.horizon_index is not None for record in result.metric_records)
    assert result.artifact_manifest
    assert any(Path(record.path).exists()
               for record in result.artifact_manifest)
    assert any(Path(record.path).name ==
               'errors.jsonl' for record in result.artifact_manifest)
    errors_path = next(
        Path(record.path) for record in result.artifact_manifest if Path(record.path).name == 'errors.jsonl')
    error_lines = [line for line in errors_path.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    assert error_lines
    assert any('AutoGluon' in line for line in error_lines)

    comparison = compare_forecasting_models_on_series(
        result,
        series_id='toy_1',
        output_dir=tmp_path / 'manual_comparison',
    )
    assert comparison.model_names
    assert comparison.artifact_manifest
    artifact_names = {
        Path(record.path).name for record in comparison.artifact_manifest}
    assert 'toy_1_history_forecast_overlay.png' in artifact_names
    assert 'toy_1_boundary_zoom.png' in artifact_names
    assert 'toy_1_forecast_delta.png' in artifact_names
    assert 'toy_1_regime_diagnostics.json' in artifact_names
    assert 'toy_1_okhs_diagnostics.json' in artifact_names
    publication_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'regime_diagnostics.csv' in publication_names
    assert 'routing_evaluation.csv' in publication_names
    assert 'routing_family_evaluation.csv' in publication_names
    assert 'routing_family_summary.csv' in publication_names
    assert 'toy_dataset_metric_distribution_mae.png' in publication_names
    assert 'toy_dataset_model_ranking_mae.png' in publication_names

    family_summary_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'routing_family_summary.csv'
    )
    family_summary = pd.read_csv(family_summary_path)
    assert 'recommended_adapter_family' in family_summary.columns
    assert 'best_adapter_family' in family_summary.columns
    assert 'family_match_rate' in family_summary.columns


def test_forecasting_suite_runs_with_new_composite_models(tmp_path: Path) -> None:
    pytest.importorskip('torch')

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(adapter_name='lagged_ridge_forecaster',
                      display_name='LaggedRidge'),
            ModelSpec(
                adapter_name='low_rank_lagged_ridge_forecaster',
                display_name='LowRankLaggedRidge',
                params={'explained_variance': 0.9},
            ),
            ModelSpec(
                adapter_name='hybrid_ensemble_forecaster',
                display_name='HybridEnsemble',
                params={'complex_branch': 'havok',
                        'complex_params': {'window_size': 8, 'rank': 2}},
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='composite_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(
        record.model_name == 'LaggedRidge' and record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == 'LowRankLaggedRidge' and record.status is RunStatus.SUCCESS for record in
               result.run_records)
    hybrid_record = next(
        record for record in result.run_records if record.model_name == 'HybridEnsemble')
    assert hybrid_record.status is RunStatus.SUCCESS
    assert hybrid_record.metadata['model_adapter_family'] == 'operator_model'
    assert hybrid_record.metadata['routing_recommendation_family'] in {
        'low_rank_linear',
        'operator_model',
        'simple_baseline',
    }
    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'toy_1_forecasting_stage_diagnostics.json' in artifact_names
    assert 'forecasting_stage_diagnostics.csv' in artifact_names
    assert 'toy_1_hybrid_ensemble_diagnostics.json' in artifact_names
    assert 'hybrid_ensemble_diagnostics.csv' in artifact_names

    stage_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_forecasting_stage_diagnostics.json'
    )
    payload = json.loads(stage_path.read_text(encoding='utf-8'))
    assert 'LaggedRidge' in payload
    assert 'LowRankLaggedRidge' in payload
    assert payload['LaggedRidge']['trajectory_transform']['window_size'] is not None
    assert payload['LowRankLaggedRidge']['rank_truncation']['selected_rank'] is not None

    hybrid_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_hybrid_ensemble_diagnostics.json'
    )
    hybrid_payload = json.loads(hybrid_path.read_text(encoding='utf-8'))
    assert 'HybridEnsemble' in hybrid_payload
    assert hybrid_payload['HybridEnsemble']['ensemble_head']['weights']
    assert 'branch_calibration' in hybrid_payload['HybridEnsemble']


def test_forecasting_suite_emits_stage_tuning_artifacts(tmp_path: Path) -> None:
    pytest.importorskip('torch')

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='lagged_ridge_forecaster',
                display_name='LaggedRidge',
                params={
                    'window_size': 10,
                    'stride': 1,
                    'alpha': 1.0,
                    'stage_tuning_runtime': {
                        'metric_name': 'rmse',
                        'max_values_per_parameter': 2,
                        'max_stage_candidates': 4,
                        'stage_updates': {
                            'trajectory_transform': {'window_size': 12},
                            'forecast_head': {'alpha': 0.5},
                        },
                    },
                },
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='stage_tuning_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    record = next(
        record for record in result.run_records if record.model_name == 'LaggedRidge')
    assert record.status is RunStatus.SUCCESS
    assert 'stage_tuning_report' in record.metadata
    assert record.metadata['stage_tuning_runtime']['enabled'] is True

    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'toy_1_forecasting_stage_tuning.json' in artifact_names
    assert 'forecasting_stage_tuning.csv' in artifact_names
    assert 'forecasting_stage_tuning_family_summary.csv' in artifact_names

    stage_tuning_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_forecasting_stage_tuning.json'
    )
    payload = json.loads(stage_tuning_path.read_text(encoding='utf-8'))
    assert 'LaggedRidge' in payload
    assert payload['LaggedRidge']['sequential_result']['best_parameters']

    family_summary_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'forecasting_stage_tuning_family_summary.csv'
    )
    family_summary = pd.read_csv(family_summary_path)
    assert 'improvement_rate' in family_summary.columns
    assert 'routing_family_match_rate' in family_summary.columns
    assert (family_summary['model_adapter_family'] == 'lagged_linear').any()


def test_forecasting_suite_runs_native_neural_bridge_with_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    class FakeNeuralBridge:
        def __init__(self, model_name, forecast_horizon, params=None):
            self.model_name = model_name
            self.forecast_horizon = int(forecast_horizon)
            self.params = dict(params or {})

        def fit(self, time_series):
            self.training_history_ = np.asarray(time_series, dtype=float)
            return self

        def predict(self, time_series=None, forecast_horizon=None):
            del time_series
            horizon = int(forecast_horizon or self.forecast_horizon)
            start = float(self.training_history_[-1]) + 0.25
            return np.linspace(start, start + horizon - 1, num=horizon)

        def get_diagnostics(self):
            return {
                'model_family': 'neural_forecaster',
                'model_name': self.model_name,
                'trajectory_transform': {
                    'kind': 'native_context_window',
                    'window_size': self.params.get('patch_len'),
                    'history_length': len(self.training_history_),
                },
                'decomposition': {},
                'rank_truncation': {},
                'forecast_head': {
                    'head_type': self.model_name,
                    'epochs': self.params.get('epochs'),
                    'batch_size': self.params.get('batch_size'),
                    'learning_rate': self.params.get('learning_rate'),
                    'forecast_horizon': self.forecast_horizon,
                },
            }

    class FakeStageTuningReport:
        def __init__(self):
            self.metadata = {
                'baseline_score': 1.2,
                'best_score': 0.7,
                'improved': True,
            }

        def to_dict(self):
            return {
                'improved': True,
                'baseline_evaluation': {
                    'metric': {'metric_name': 'rmse', 'metric_value': 1.2},
                    'parameters': {'patch_len': 8, 'epochs': 2},
                    'diagnostics': {'model_family': 'neural_forecaster'},
                },
                'best_evaluation': {
                    'metric': {'metric_name': 'rmse', 'metric_value': 0.7},
                    'parameters': {'patch_len': 12, 'epochs': 4},
                    'diagnostics': {'model_family': 'neural_forecaster'},
                },
                'sequential_result': {
                    'best_parameters': {'patch_len': 12, 'epochs': 4},
                    'stage_history': [
                        {'stage': 'trajectory_transform',
                            'applied_parameters': {'patch_len': 12}},
                        {'stage': 'forecast_head',
                            'applied_parameters': {'epochs': 4}},
                    ],
                },
                'metadata': dict(self.metadata),
            }

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_neural_forecast_head_on_series',
        lambda model_name, time_series, forecast_horizon, params=None: type(
            'FakeNeuralRunResult',
            (),
            {
                'forecast': tuple(
                    FakeNeuralBridge(
                        model_name=model_name,
                        forecast_horizon=forecast_horizon,
                        params=params,
                    ).fit(time_series).predict(time_series)
                ),
                'diagnostics': FakeNeuralBridge(
                    model_name=model_name,
                    forecast_horizon=forecast_horizon,
                    params=params,
                ).fit(time_series).get_diagnostics(),
            },
        )(),
    )
    monkeypatch.setattr('benchmark.industrial.forecasting.torch', object())
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_forecasting_stage_tuning_on_series',
        lambda *args, **kwargs: FakeStageTuningReport(),
    )

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='patch_tst_model',
                display_name='PatchTST',
                params={
                    'patch_len': 12,
                    'epochs': 2,
                    'batch_size': 4,
                    'learning_rate': 1e-3,
                    'stage_tuning_runtime': {
                        'metric_name': 'rmse',
                        'max_values_per_parameter': 2,
                        'max_stage_candidates': 4,
                    },
                },
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='neural_bridge_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    record = next(
        record for record in result.run_records if record.model_name == 'PatchTST')
    assert record.status is RunStatus.SUCCESS
    assert record.metadata['model_adapter_family'] == 'neural_forecaster'
    assert record.metadata['routing_recommendation_family'] in {
        'periodic_linear',
        'operator_model',
        'simple_baseline',
        'neural_forecaster',
        'lagged_linear',
        'low_rank_linear',
    }
    assert record.metadata['forecast_head']['head_type'] == 'patch_tst_model'
    assert record.metadata['stage_tuning_runtime']['enabled'] is True
    assert record.metadata['stage_tuning_report']['sequential_result']['best_parameters']['patch_len'] == 12

    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'toy_1_forecasting_stage_diagnostics.json' in artifact_names
    assert 'toy_1_forecasting_stage_tuning.json' in artifact_names
    assert 'forecasting_stage_diagnostics.csv' in artifact_names
    assert 'forecasting_stage_tuning.csv' in artifact_names
    assert 'forecasting_stage_tuning_family_summary.csv' in artifact_names

    stage_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_forecasting_stage_diagnostics.json'
    )
    stage_payload = json.loads(stage_path.read_text(encoding='utf-8'))
    assert 'PatchTST' in stage_payload
    assert stage_payload['PatchTST']['forecast_head']['head_type'] == 'patch_tst_model'
    assert stage_payload['PatchTST']['trajectory_transform']['window_size'] == 12

    tuning_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_forecasting_stage_tuning.json'
    )
    tuning_payload = json.loads(tuning_path.read_text(encoding='utf-8'))
    assert 'PatchTST' in tuning_payload
    assert tuning_payload['PatchTST']['sequential_result']['stage_history'][0]['stage'] == 'trajectory_transform'

    family_summary_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'forecasting_stage_tuning_family_summary.csv'
    )
    family_summary = pd.read_csv(family_summary_path)
    assert (family_summary['model_adapter_family']
            == 'neural_forecaster').any()


def test_forecasting_suite_emits_havok_event_artifacts(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='switch_dataset',
                subset='daily',
                adapter_options={
                    'records': _switching_records(),
                    'forecast_horizon': 6,
                    'seasonal_period': 7,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='havok',
                display_name='HAVOK',
                params={'window_size': 14, 'rank': 4,
                        'forcing_threshold_scale': 0.75},
            ),
            ModelSpec(adapter_name='naive_last_value',
                      display_name='NaiveLastValue'),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='switch_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    havok_record = next(
        record for record in result.run_records if record.model_name == 'HAVOK')
    assert havok_record.status is RunStatus.SUCCESS
    assert 'forecast_forcing_mask' in havok_record.metadata
    assert 'regime_diagnostics' in havok_record.metadata
    assert havok_record.metadata['routing_recommendation']['primary_adapter'] == 'havok'
    assert any(
        record.model_name == 'HAVOK' and record.metric_name in {
            'mae_active', 'mae_calm'}
        for record in result.metric_records
    )
    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'switch_1_havok_diagnostics.json' in artifact_names
    assert 'switch_1_havok_havok_forcing_timeline.png' in artifact_names
    assert 'switch_1_havok_havok_event_overlay.png' in artifact_names
    assert 'switch_1_forecasting_stage_diagnostics.json' in artifact_names

    stage_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'switch_1_forecasting_stage_diagnostics.json'
    )
    stage_payload = json.loads(stage_path.read_text(encoding='utf-8'))
    assert 'HAVOK' in stage_payload
    assert stage_payload['HAVOK']['forecast_head']['head_type'] == 'havok_head'
    assert stage_payload['HAVOK']['trajectory_transform']['kind'] == 'hankel'


def test_forecasting_suite_propagates_okhs_anti_smoothing_diagnostics(tmp_path: Path, monkeypatch) -> None:
    class FakeOKHSForecaster:
        def __init__(self, **kwargs):
            self.forecast_horizon = int(kwargs['forecast_horizon'])
            self.resolved_q_ = float(kwargs.get('q', 0.7))

        def fit(self, time_series, window_size=20):
            del time_series, window_size
            return self

        def predict(self, time_series=None):
            del time_series
            return np.linspace(1.0, 0.8, num=self.forecast_horizon)

        def get_optimization_info(self):
            return {
                'fdmd_prediction_diagnostics': {
                    'anti_smoothing_diagnostics': {
                        'collapse_detected': True,
                        'correction_applied': True,
                        'forecast_amplitude_before': 0.01,
                        'forecast_amplitude_after': 0.15,
                    }
                }
            }

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.OKHSForecaster', FakeOKHSForecaster)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='okhs',
                display_name='OKHS DMD',
                params={'method': 'dmd', 'window_size': 8,
                        'n_modes': 2, 'q': 0.9},
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='okhs_diag_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    okhs_record = next(
        record for record in result.run_records if record.model_name == 'OKHS DMD')
    anti_smoothing = okhs_record.metadata['fdmd_prediction_diagnostics']['anti_smoothing_diagnostics']
    assert anti_smoothing['collapse_detected'] is True
    assert anti_smoothing['correction_applied'] is True


def test_forecasting_suite_emits_okhs_fdmd_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_okhs_fdmd_forecaster_on_series',
        lambda time_series, forecast_horizon, params=None: type(
            'FakeOKHSRunResult',
            (),
            {
                'forecast': tuple(np.linspace(1.0, 1.3, num=int(forecast_horizon))),
                'diagnostics': {
                    'model_family': 'operator_model',
                    'model_name': 'okhs_fdmd_forecaster',
                    'trajectory_transform': {
                        'window_policy': 'adaptive_cycle_aware',
                        'resolved_window_size': int((params or {}).get('window_size', 8)),
                        'expected_overlap_ratio': 0.8,
                        'effective_stride': 2,
                        'effective_trajectory_count': 10,
                        'trajectory_matrix_shape_before': (10, int((params or {}).get('window_size', 8))),
                        'trajectory_matrix_shape_after': (10, int((params or {}).get('window_size', 8))),
                    },
                    'decomposition': {
                        'representation_policy': 'projected',
                        'projected_shape': (10, 4),
                        'basis_shape': (int((params or {}).get('window_size', 8)), 4),
                        'decode_supported': True,
                        'decode_reconstruction_error': 0.01,
                        'latent_window_size': 4,
                        'latent_stride': 2,
                    },
                    'rank_truncation': {
                        'trajectory_rank_policy': 'explained_dispersion',
                        'selected_rank': 4,
                        'raw_selected_rank': 3,
                        'explained_variance_retained': 0.96,
                        'compression_ratio': 0.5,
                    },
                    'forecast_head': {
                        'q': float((params or {}).get('q', 0.7)),
                        'q_policy': 'fixed',
                        'mode_selection_policy': 'energy',
                        'prediction_mode_selection_policy': 'adaptive_tail_energy',
                        'boundary_alignment_policy': 'tapered_offset',
                        'prediction_stability_threshold': 0.03,
                        'fit_diagnostics': {
                            'resolved_n_modes': 5,
                        },
                        'prediction_diagnostics': {
                            'n_selected_prediction_modes': 3,
                            'boundary_discontinuity_abs_mean': 0.25,
                        },
                        'anti_smoothing': {
                            'collapse_detected': False,
                            'correction_applied': False,
                            'collapse_resolved': True,
                        },
                    },
                },
            },
        )(),
    )

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='okhs_fdmd_forecaster',
                display_name='OKHS FDMD',
                params={'window_size': 8, 'q': 0.85},
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='okhs_fdmd_stage_suite',
                         primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    artifact_names = {
        Path(record.path).name for record in result.artifact_manifest}
    assert 'toy_1_okhs_fdmd_stage_diagnostics.json' in artifact_names
    assert 'okhs_fdmd_stage_diagnostics.csv' in artifact_names

    stage_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'toy_1_okhs_fdmd_stage_diagnostics.json'
    )
    payload = json.loads(stage_path.read_text(encoding='utf-8'))
    assert 'OKHS FDMD' in payload
    assert payload['OKHS FDMD']['trajectory_transform']['resolved_window_size'] == 8
    assert payload['OKHS FDMD']['decomposition']['representation_policy'] == 'projected'
    assert payload['OKHS FDMD']['forecast_head']['fit_diagnostics']['resolved_n_modes'] == 5


def test_okhs_fdmd_adapter_normalizes_runtime_spec_before_execution(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def _fake_build_spec(*, forecast_horizon, params=None, series_length=None):
        captured['requested_forecast_horizon'] = int(forecast_horizon)
        captured['requested_window_size'] = int(
            (params or {}).get('window_size', 0))
        captured['series_length'] = int(series_length)
        return type(
            'FakeOKHSSpec',
            (),
            {
                'params': {
                    **dict(params or {}),
                    'window_size': min(max(int((params or {}).get('window_size', 20)), 4), int(series_length) - 1),
                },
            },
        )()

    monkeypatch.setattr(
        'benchmark.industrial.forecasting.build_okhs_fdmd_spec', _fake_build_spec)
    monkeypatch.setattr(
        'benchmark.industrial.forecasting.run_okhs_fdmd_forecaster_on_series',
        lambda time_series, forecast_horizon, params=None: type(
            'FakeOKHSRunResult',
            (),
            {
                'forecast': tuple(np.linspace(1.0, 1.2, num=int(forecast_horizon))),
                'diagnostics': {
                    'model_family': 'operator_model',
                    'model_name': 'okhs_fdmd_forecaster',
                    'trajectory_transform': {'resolved_window_size': int((params or {}).get('window_size', 0))},
                    'decomposition': {},
                    'rank_truncation': {},
                    'forecast_head': {},
                },
            },
        )(),
    )

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='okhs_window_clip',
                subset='monthly',
                adapter_options={
                    'records': [{
                        'series_id': 'clip_case',
                        'train_values': list(np.arange(12, dtype=float)),
                        'test_values': [12.0, 13.0, 14.0, 15.0],
                    }],
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='okhs_fdmd_forecaster',
                display_name='OKHS FDMD Clipped',
                params={'window_size': 100, 'q': 0.7, 'n_modes': 3},
            ),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='okhs_clip_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    record = next(
        record for record in result.run_records if record.model_name == 'OKHS FDMD Clipped')
    assert captured['requested_forecast_horizon'] == 4
    assert captured['requested_window_size'] == 100
    assert captured['series_length'] == 12
    assert record.metadata['window_size'] == 11
    assert record.metadata['trajectory_transform']['resolved_window_size'] == 11


def test_forecasting_publication_pack_falls_back_without_tabulate(tmp_path: Path, monkeypatch) -> None:
    original_to_markdown = pd.DataFrame.to_markdown

    def broken_to_markdown(self, *args, **kwargs):
        del self, args, kwargs
        raise ImportError("Missing optional dependency 'tabulate'.")

    monkeypatch.setattr(pd.DataFrame, 'to_markdown', broken_to_markdown)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_dataset',
                subset='monthly',
                adapter_options={
                    'records': _toy_records(),
                    'forecast_horizon': 4,
                    'seasonal_period': 4,
                },
            ),
        ),
        models=(
            ModelSpec(adapter_name='naive_last_value',
                      display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average',
                      display_name='MovingAverage', params={'window_size': 3}),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_suite_no_tabulate',
                         primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert result.artifact_manifest
    summary_path = next(
        Path(record.path) for record in result.artifact_manifest
        if Path(record.path).name == 'summary.md'
    )
    summary_text = summary_path.read_text(encoding='utf-8')
    assert '# Forecasting Benchmark Summary' in summary_text
    assert '| benchmark' in summary_text or '| model_name' in summary_text

    monkeypatch.setattr(pd.DataFrame, 'to_markdown', original_to_markdown)


def test_forecasting_suite_runs_on_local_m4_subset(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='m4',
                dataset_name='m4_daily_local',
                subset='daily',
                sample_size=1,
                random_seed=7,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=(
            ModelSpec(adapter_name='naive_last_value',
                      display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average',
                      display_name='MovingAverage', params={'window_size': 3}),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='real_local_m4', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.metric_name ==
               'mae' and record.horizon_index is None for record in result.metric_records)
    assert any(record.metric_name ==
               'mae' and record.horizon_index is not None for record in result.metric_records)


def test_forecasting_suite_loads_long_local_m4_layout(tmp_path: Path) -> None:
    local_dir = tmp_path / 'm4'
    local_dir.mkdir()
    (local_dir / 'M4Monthly.csv').write_text(
        '\n'.join([
            'datetime,value,label',
            '2020-01-31,1.0,M1',
            '2020-02-29,2.0,M1',
            '2020-03-31,3.0,M1',
            '2020-04-30,4.0,M1',
            '2020-05-31,5.0,M1',
            '2020-06-30,6.0,M1',
            '2020-07-31,7.0,M1',
            '2020-08-31,8.0,M1',
            '2020-09-30,9.0,M1',
            '2020-10-31,10.0,M1',
            '2020-11-30,11.0,M1',
            '2020-12-31,12.0,M1',
            '2021-01-31,13.0,M1',
            '2021-02-28,14.0,M1',
            '2021-03-31,15.0,M1',
            '2021-04-30,16.0,M1',
            '2021-05-31,17.0,M1',
            '2021-06-30,18.0,M1',
            '2021-07-31,19.0,M1',
            '2021-08-31,20.0,M1',
        ]),
        encoding='utf-8',
    )
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='m4',
                dataset_name='m4_monthly_long',
                subset='monthly',
                adapter_options={'use_local_files': True,
                                 'local_csv_dir': str(local_dir)},
            ),
        ),
        models=(ModelSpec(adapter_name='naive_last_value',
                display_name='NaiveLastValue'),),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='long_local_m4', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert result.run_records[0].metadata['series_metadata']['split_provenance'] == 'local_m4_long_tail_holdout'


def test_forecasting_suite_runs_on_local_monash_subset(tmp_path: Path) -> None:
    config = BenchmarkSuiteConfig(
        task_type=TaskType.FORECASTING,
        datasets=(
            DatasetSpec(
                benchmark='monash',
                dataset_name='Bitcoin',
                subset='daily',
                sample_size=1,
                random_seed=11,
                adapter_options={'use_local_files': True},
            ),
        ),
        models=(
            ModelSpec(adapter_name='naive_last_value',
                      display_name='NaiveLastValue'),
            ModelSpec(adapter_name='naive_mean', display_name='NaiveMean'),
        ),
        artifact_spec=ArtifactSpec(
            output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='real_local_monash', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.metric_name ==
               'smape' for record in result.metric_records)
    assert any(record.metric_name == 'owa' for record in result.metric_records)
