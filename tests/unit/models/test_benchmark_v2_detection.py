from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.v2.core import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
)
from benchmark.v2.detection import (
    MPSIAdapter,
    compute_detection_metric,
    run_anomaly_detection_suite,
)
from benchmark.v2.presets import build_local_skab_suite_config as preset_skab_config


def _toy_detection_record() -> dict:
    """Return a tiny synthetic record in legacy pair payload format."""
    values = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [0.9, 1.0],
            [2.0, 2.1],
            [2.2, 2.0],
        ]
    )
    target = np.array([0, 0, 0, 1, 1, 0])
    train_values = values[:3]
    test_values = values[3:]
    test_target = target[3:]
    return {
        'series_id': 'toy_series',
        'train_values': train_values,
        'test_values': test_values,
        'test_target': test_target,
    }


def test_compute_detection_metric_accuracy() -> None:
    """Smoke-check: accuracy metric follows expected classification behavior."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert compute_detection_metric('accuracy', y_true, y_pred) == 0.75


def test_in_memory_detection_suite_runs(tmp_path: Path) -> None:
    """End-to-end smoke test for in-memory detection suite execution."""
    config = BenchmarkSuiteConfig(
        task_type=TaskType.ANOMALY_DETECTION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory',
                dataset_name='toy_detection',
                adapter_options={'records': [_toy_detection_record()]},
            ),
        ),
        models=(
            ModelSpec(adapter_name='constant_zero', display_name='ConstantZero'),
            ModelSpec(adapter_name='feature_iforest_detector', display_name='IForest'),
        ),
        metrics=('accuracy', 'f1_macro'),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=False),
        run_spec=RunSpec(run_name='toy_detection', primary_metric='accuracy', show_progress=False),
    )
    result = run_anomaly_detection_suite(config)
    assert result.run_records
    assert all(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert result.prediction_records
    assert result.metric_records


def test_mpsi_stub_requires_records() -> None:
    """MPSI adapter must fail fast when stub payload is absent."""
    spec = DatasetSpec(benchmark='mpsi', dataset_name='mpsi')
    try:
        MPSIAdapter().load_series(spec)
    except Exception as exc:
        assert 'records' in str(exc).lower()
    else:
        raise AssertionError('Expected MPSI stub to require in-memory records.')


def test_mpsi_stub_loads_in_memory_records() -> None:
    """MPSI stub should accept in-memory records during phase-1 wiring."""
    spec = DatasetSpec(
        benchmark='mpsi',
        dataset_name='mpsi_stub',
        adapter_options={'records': [_toy_detection_record()]},
    )
    records = MPSIAdapter().load_series(spec)
    assert len(records) == 1
    assert records[0].series_id == 'toy_series'


def test_skab_preset_smoke_runs_one_series(tmp_path: Path) -> None:
    """SKAB preset smoke test on one small series for quick CI feedback."""
    config = preset_skab_config(
        output_dir=tmp_path,
        persist_on_run=False,
        models=(ModelSpec(adapter_name='constant_zero', display_name='ConstantZero'),),
    )
    dataset_spec = config.datasets[0]
    config = BenchmarkSuiteConfig(
        task_type=config.task_type,
        datasets=(
            DatasetSpec(
                benchmark=dataset_spec.benchmark,
                dataset_name=dataset_spec.dataset_name,
                subset=dataset_spec.subset,
                sample_size=1,
                series_ids=('1',),
                adapter_options={
                    'folder': 'valve1',
                    'split_mode': 'legacy_pair',
                    'train_data_size': 'anomaly-free',
                },
            ),
        ),
        models=config.models,
        metrics=config.metrics,
        artifact_spec=config.artifact_spec,
        run_spec=RunSpec(
            run_name='skab_smoke',
            primary_metric='accuracy',
            show_progress=False,
        ),
    )
    result = run_anomaly_detection_suite(config)
    assert result.run_records
    assert result.run_records[0].status is RunStatus.SUCCESS
    assert 'accuracy' in result.run_records[0].metrics_summary
