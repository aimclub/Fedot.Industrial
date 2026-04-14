from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.benchmark_TSF import BenchmarkTSF
from benchmark.v2.api import compare_forecasting_models_on_series, run_forecasting_benchmark_suite
from benchmark.v2.core import ArtifactSpec, BenchmarkSuiteConfig, DatasetSpec, ModelSpec, RunSpec, RunStatus, TaskType
from benchmark.v2.forecasting import build_dataset_adapter, build_model_adapter


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
        ModelSpec(adapter_name='lagged_forecaster', display_name='lagged_forecaster')
    )
    ssa_model = build_model_adapter(
        ModelSpec(adapter_name='ssa_forecaster', display_name='ssa_forecaster')
    )

    assert lagged_model.name == 'lagged_forecaster'
    assert ssa_model.name == 'ssa_forecaster'


def test_build_model_adapter_supports_new_composite_forecasters() -> None:
    lagged_ridge_model = build_model_adapter(
        ModelSpec(adapter_name='lagged_ridge_forecaster', display_name='lagged_ridge_forecaster')
    )
    low_rank_model = build_model_adapter(
        ModelSpec(adapter_name='low_rank_lagged_ridge_forecaster', display_name='low_rank_lagged_ridge_forecaster')
    )
    hybrid_model = build_model_adapter(
        ModelSpec(adapter_name='hybrid_ensemble_forecaster', display_name='hybrid_ensemble_forecaster')
    )
    okhs_fdmd_model = build_model_adapter(
        ModelSpec(adapter_name='okhs_fdmd_forecaster', display_name='okhs_fdmd_forecaster')
    )

    assert lagged_ridge_model.name == 'lagged_ridge_forecaster'
    assert low_rank_model.name == 'low_rank_lagged_ridge_forecaster'
    assert hybrid_model.name == 'hybrid_ensemble_forecaster'
    assert okhs_fdmd_model.name == 'okhs_fdmd_forecaster'


def test_build_model_adapter_supports_canonical_mssa_and_havok_names() -> None:
    mssa_model = build_model_adapter(
        ModelSpec(adapter_name='mssa_forecaster', display_name='mssa_forecaster')
    )
    havok_model = build_model_adapter(
        ModelSpec(adapter_name='havok_forecaster', display_name='havok_forecaster')
    )

    assert mssa_model.name == 'mssa_forecaster'
    assert havok_model.name == 'havok_forecaster'


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
    assert all(record.metadata['split_provenance'] == 'local_m4_train_test_csv' for record in records)


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
    assert all(record.metadata['split_provenance'] == 'local_monash_csv' for record in records)
    assert all(record.metadata['source_file'] == 'MonashBitcoin_30.csv' for record in records)


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
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average', display_name='MovingAverage', params={'window_size': 3}),
            ModelSpec(adapter_name='mssa', display_name='mSSA', params={'window_size': 6, 'rank': 2}),
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
            ModelSpec(adapter_name='autogluon', display_name='AutoGluon', optional=True),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert result.aggregate_report.primary_metric == 'mae'
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(
        record.model_name == 'OKHS Direct' and record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == 'mSSA' and record.status is RunStatus.SUCCESS for record in result.run_records)
    okhs_record = next(record for record in result.run_records if record.model_name == 'OKHS Direct')
    assert okhs_record.metadata['method'] == 'direct'
    assert 'regime_diagnostics' in okhs_record.metadata
    assert 'routing_recommendation' in okhs_record.metadata
    assert any(record.horizon_index is None for record in result.metric_records)
    assert any(record.horizon_index is not None for record in result.metric_records)
    assert result.artifact_manifest
    assert any(Path(record.path).exists() for record in result.artifact_manifest)
    assert any(Path(record.path).name == 'errors.jsonl' for record in result.artifact_manifest)
    errors_path = next(
        Path(record.path) for record in result.artifact_manifest if Path(record.path).name == 'errors.jsonl')
    error_lines = [line for line in errors_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert error_lines
    assert any('AutoGluon' in line for line in error_lines)

    comparison = compare_forecasting_models_on_series(
        result,
        series_id='toy_1',
        output_dir=tmp_path / 'manual_comparison',
    )
    assert comparison.model_names
    assert comparison.artifact_manifest
    artifact_names = {Path(record.path).name for record in comparison.artifact_manifest}
    assert 'toy_1_history_forecast_overlay.png' in artifact_names
    assert 'toy_1_boundary_zoom.png' in artifact_names
    assert 'toy_1_forecast_delta.png' in artifact_names
    assert 'toy_1_regime_diagnostics.json' in artifact_names
    assert 'toy_1_okhs_diagnostics.json' in artifact_names
    publication_names = {Path(record.path).name for record in result.artifact_manifest}
    assert 'regime_diagnostics.csv' in publication_names
    assert 'routing_evaluation.csv' in publication_names
    assert 'toy_dataset_metric_distribution_mae.png' in publication_names
    assert 'toy_dataset_model_ranking_mae.png' in publication_names


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
            ModelSpec(adapter_name='lagged_ridge_forecaster', display_name='LaggedRidge'),
            ModelSpec(
                adapter_name='low_rank_lagged_ridge_forecaster',
                display_name='LowRankLaggedRidge',
                params={'explained_variance': 0.9},
            ),
            ModelSpec(
                adapter_name='hybrid_ensemble_forecaster',
                display_name='HybridEnsemble',
                params={'complex_branch': 'havok', 'complex_params': {'window_size': 8, 'rank': 2}},
            ),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='composite_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(
        record.model_name == 'LaggedRidge' and record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == 'LowRankLaggedRidge' and record.status is RunStatus.SUCCESS for record in
               result.run_records)
    hybrid_record = next(record for record in result.run_records if record.model_name == 'HybridEnsemble')
    assert hybrid_record.status is RunStatus.SUCCESS
    assert hybrid_record.metadata['model_adapter_family'] == 'operator_model'
    assert hybrid_record.metadata['routing_recommendation_family'] in {
        'low_rank_linear',
        'operator_model',
        'simple_baseline',
    }
    artifact_names = {Path(record.path).name for record in result.artifact_manifest}
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
                params={'window_size': 14, 'rank': 4, 'forcing_threshold_scale': 0.75},
            ),
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='switch_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    havok_record = next(record for record in result.run_records if record.model_name == 'HAVOK')
    assert havok_record.status is RunStatus.SUCCESS
    assert 'forecast_forcing_mask' in havok_record.metadata
    assert 'regime_diagnostics' in havok_record.metadata
    assert havok_record.metadata['routing_recommendation']['primary_adapter'] == 'havok'
    assert any(
        record.model_name == 'HAVOK' and record.metric_name in {'mae_active', 'mae_calm'}
        for record in result.metric_records
    )
    artifact_names = {Path(record.path).name for record in result.artifact_manifest}
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

    monkeypatch.setattr('benchmark.v2.forecasting.OKHSForecaster', FakeOKHSForecaster)

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
                params={'method': 'dmd', 'window_size': 8, 'n_modes': 2, 'q': 0.9},
            ),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='okhs_diag_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    okhs_record = next(record for record in result.run_records if record.model_name == 'OKHS DMD')
    anti_smoothing = okhs_record.metadata['fdmd_prediction_diagnostics']['anti_smoothing_diagnostics']
    assert anti_smoothing['collapse_detected'] is True
    assert anti_smoothing['correction_applied'] is True


def test_forecasting_suite_emits_okhs_fdmd_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    class FakeOKHSFDMDForecaster:
        def __init__(self, forecast_horizon, **kwargs):
            self.forecast_horizon = int(forecast_horizon)
            self.window_size = int(kwargs.get('window_size', 8))
            self.q = float(kwargs.get('q', 0.7))

        def fit(self, time_series):
            self.training_history_ = np.asarray(time_series, dtype=float)
            return self

        def predict(self, time_series=None, forecast_horizon=None):
            del time_series
            horizon = int(forecast_horizon or self.forecast_horizon)
            return np.linspace(1.0, 1.3, num=horizon)

        def get_diagnostics(self):
            return {
                'model_family': 'operator_model',
                'model_name': 'okhs_fdmd_forecaster',
                'trajectory_transform': {
                    'window_policy': 'adaptive_cycle_aware',
                    'resolved_window_size': self.window_size,
                    'expected_overlap_ratio': 0.8,
                    'effective_stride': 2,
                    'effective_trajectory_count': 10,
                    'trajectory_matrix_shape_before': (10, self.window_size),
                    'trajectory_matrix_shape_after': (10, self.window_size),
                },
                'decomposition': {
                    'representation_policy': 'projected',
                    'projected_shape': (10, 4),
                    'basis_shape': (self.window_size, 4),
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
                    'q': self.q,
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
            }

    monkeypatch.setattr('benchmark.v2.forecasting.OKHSFDMDForecaster', FakeOKHSFDMDForecaster)

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
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='okhs_fdmd_stage_suite', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    artifact_names = {Path(record.path).name for record in result.artifact_manifest}
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
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average', display_name='MovingAverage', params={'window_size': 3}),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='toy_suite_no_tabulate', primary_metric='mae'),
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
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='moving_average', display_name='MovingAverage', params={'window_size': 3}),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='real_local_m4', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.metric_name == 'mae' and record.horizon_index is None for record in result.metric_records)
    assert any(record.metric_name == 'mae' and record.horizon_index is not None for record in result.metric_records)


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
            ModelSpec(adapter_name='naive_last_value', display_name='NaiveLastValue'),
            ModelSpec(adapter_name='naive_mean', display_name='NaiveMean'),
        ),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='real_local_monash', primary_metric='mae'),
    )

    result = run_forecasting_benchmark_suite(config)

    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.metric_name == 'smape' for record in result.metric_records)
    assert any(record.metric_name == 'owa' for record in result.metric_records)


def test_legacy_benchmark_tsf_can_delegate_to_v2(tmp_path: Path) -> None:
    benchmark = BenchmarkTSF(
        experiment_setup={
            'use_benchmark_v2': True,
            'output_dir': str(tmp_path),
            'dataset_specs': [
                {
                    'benchmark': 'in_memory',
                    'dataset_name': 'toy_dataset',
                    'subset': 'monthly',
                    'adapter_options': {
                        'records': _toy_records(),
                        'forecast_horizon': 4,
                        'seasonal_period': 4,
                    },
                }
            ],
            'model_specs': [
                {'adapter_name': 'naive_last_value', 'display_name': 'NaiveLastValue'},
            ],
        },
        custom_datasets=[],
    )

    result = benchmark.run()

    assert result.config.task_type is TaskType.FORECASTING
    assert any(record.model_name == 'NaiveLastValue' for record in result.run_records)
