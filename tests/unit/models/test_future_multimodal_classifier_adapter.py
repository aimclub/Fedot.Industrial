"""Smoke tests for FUTURE multimodal and MiniRocketRidge classification adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark.industrial import (
    ArtifactSpec,
    BenchmarkSuiteConfig,
    DatasetSpec,
    ModelSpec,
    RunSpec,
    RunStatus,
    TaskType,
    run_tsc_benchmark_suite,
)
from benchmark.industrial.errors import BenchmarkClassificationError
from benchmark.industrial.experiments.presets import run_local_benchmark_preset
from benchmark.industrial.models.classification import (
    FutureFusionClassifierAdapter,
    FutureMultimodalClassifierAdapter,
    MiniRocketRidgeClassifierAdapter,
    build_classification_model,
)


def _synthetic_tsc(
    *,
    n_train: int = 20,
    n_test: int = 6,
    seq_len: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_x = rng.normal(size=(n_train, seq_len))
    train_y = np.asarray(['a', 'b'] * (n_train // 2), dtype=object)
    test_x = rng.normal(size=(n_test, seq_len))
    return train_x, train_y, test_x


def test_build_future_multimodal_classifier_adapter():
    model = build_classification_model(
        ModelSpec(
            adapter_name='future_multimodal_classifier',
            display_name='FutureConcat',
            optional=True,
            params={
                'fusion_method': 'concat',
                'd_model': 16,
                'training': {'epochs': 1, 'batch_size': 8, 'device': 'cpu'},
            },
        )
    )
    assert isinstance(model, FutureMultimodalClassifierAdapter)
    assert FutureFusionClassifierAdapter is FutureMultimodalClassifierAdapter
    status, message = model.availability()
    assert status is RunStatus.SUCCESS
    assert message == 'ready'


def test_build_minirocket_ridge_classifier_adapter():
    model = build_classification_model(
        ModelSpec(
            adapter_name='minirocket_ridge_classifier',
            display_name='MiniRocketRidge',
            params={'num_features': 168, 'seed': 0},
        )
    )
    assert isinstance(model, MiniRocketRidgeClassifierAdapter)
    status, message = model.availability()
    assert status is RunStatus.SUCCESS
    assert message == 'ready'


def test_build_classification_model_unknown_adapter_lists_registry():
    with pytest.raises(BenchmarkClassificationError, match='Available adapters'):
        build_classification_model(
            ModelSpec(adapter_name='unknown_adapter', display_name='Unknown')
        )


def test_future_multimodal_classifier_adapter_fit_predict_smoke():
    train_x, train_y, test_x = _synthetic_tsc()

    model = FutureMultimodalClassifierAdapter(
        name='FutureConcat',
        params={
            'fusion_method': 'concat',
            'd_model': 16,
            'modalities': ['raw'],
            'output_diagnostics': True,
            'training': {
                'epochs': 1,
                'train_batch_size': 8,
                'val_batch_size': 8,
                'validation_fraction': 0.2,
                'patience': 1,
                'device': 'cpu',
                'seed': 0,
            },
            'preparation': {
                'torch_device': 'cpu',
                'transformation_config': {
                    'raw': {'per_sample_z_normalize': False},
                },
                'normalization_config': {},
            },
        },
    )
    status, _ = model.availability()
    assert status is RunStatus.SUCCESS

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    assert predictions.shape == (6,)
    assert set(predictions.tolist()).issubset({'a', 'b'})

    artifacts = model.export_artifacts()
    assert artifacts['adapter'] == 'future_multimodal_classifier'
    assert artifacts['train_duration_s'] > 0
    assert artifacts['training_history']['best_epoch'] >= 1
    assert artifacts['output_diagnostics'] is True
    assert 'diagnostics' in artifacts
    assert 'active_modalities' in artifacts['diagnostics']


def test_minirocket_ridge_classifier_adapter_fit_predict_smoke():
    train_x, train_y, test_x = _synthetic_tsc(n_train=24, n_test=8, seq_len=48)

    model = MiniRocketRidgeClassifierAdapter(
        name='MiniRocketRidge',
        params={'num_features': 168, 'seed': 0, 'device': 'cpu'},
    )
    status, _ = model.availability()
    assert status is RunStatus.SUCCESS

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    assert predictions.shape == (8,)
    assert set(predictions.tolist()).issubset({'a', 'b'})

    artifacts = model.export_artifacts()
    assert artifacts['adapter'] == 'minirocket_ridge_classifier'
    assert artifacts['num_features'] == model.num_features_
    assert artifacts['ridge']['coef_shape']


def test_future_multimodal_adapter_in_memory_tsc_suite(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    train_features = rng.normal(size=(16, 24))
    train_target = np.asarray(['a', 'b'] * 8, dtype=object)
    test_features = rng.normal(size=(4, 24))
    test_target = np.asarray(['a', 'b', 'a', 'b'], dtype=object)

    config = BenchmarkSuiteConfig(
        task_type=TaskType.TS_CLASSIFICATION,
        datasets=(
            DatasetSpec(
                benchmark='in_memory_tsc',
                dataset_name='future_toy_tsc',
                adapter_options={
                    'record': {
                        'train_features': train_features,
                        'train_target': train_target,
                        'test_features': test_features,
                        'test_target': test_target,
                    }
                },
            ),
        ),
        models=(
            ModelSpec(
                adapter_name='future_multimodal_classifier',
                display_name='FutureConcat',
                optional=True,
                params={
                    'fusion_method': 'concat',
                    'd_model': 16,
                    'modalities': ['raw'],
                    'training': {
                        'epochs': 1,
                        'batch_size': 8,
                        'device': 'cpu',
                        'seed': 0,
                    },
                    'preparation': {
                        'torch_device': 'cpu',
                        'transformation_config': {
                            'raw': {'per_sample_z_normalize': False},
                        },
                        'normalization_config': {},
                    },
                },
            ),
        ),
        metrics=('accuracy', 'balanced_accuracy', 'f1_macro'),
        artifact_spec=ArtifactSpec(output_dir=str(tmp_path), persist_on_run=True),
        run_spec=RunSpec(run_name='future_toy_tsc', primary_metric='accuracy'),
    )

    result = run_tsc_benchmark_suite(config)
    assert any(record.status is RunStatus.SUCCESS for record in result.run_records)
    assert any(record.model_name == 'FutureConcat' for record in result.run_records)
    assert result.artifact_manifest
    assert any(Path(item.path).exists() for item in result.artifact_manifest)


def test_run_local_benchmark_preset_smoke_minirocket_ridge() -> None:
    result = run_local_benchmark_preset(
        'ucr',
        dataset_name='Lightning7',
        persist_on_run=False,
        models=(
            ModelSpec(
                adapter_name='minirocket_ridge_classifier',
                display_name='MiniRocketRidge',
                params={'num_features': 168, 'seed': 0, 'device': 'cpu'},
            ),
        ),
    )

    assert result.config.task_type is TaskType.TS_CLASSIFICATION
    assert any(record.status.value == 'success' for record in result.run_records)
    assert any(record.model_name == 'MiniRocketRidge' for record in result.run_records)
