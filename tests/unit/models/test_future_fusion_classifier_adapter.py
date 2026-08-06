"""Smoke tests for FUTURE benchmark classification adapter."""

from __future__ import annotations

import numpy as np

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.models.classification import (
    FutureFusionClassifierAdapter,
    build_classification_model,
)


def test_build_future_fusion_classifier_adapter():
    model = build_classification_model(
        ModelSpec(
            adapter_name='future_fusion_classifier',
            display_name='FutureConcat',
            optional=True,
            params={
                'fusion_method': 'concat',
                'd_model': 16,
                'training': {'epochs': 1, 'batch_size': 8, 'device': 'cpu'},
            },
        )
    )
    assert isinstance(model, FutureFusionClassifierAdapter)
    status, message = model.availability()
    assert status is RunStatus.SUCCESS
    assert message == 'ready'


def test_future_fusion_classifier_adapter_fit_predict_smoke():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(20, 32))
    train_y = np.asarray(['a', 'b'] * 10, dtype=object)
    test_x = rng.normal(size=(6, 32))

    model = FutureFusionClassifierAdapter(
        name='FutureConcat',
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
    )
    status, _ = model.availability()
    assert status is RunStatus.SUCCESS

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    assert predictions.shape == (6,)
    assert set(predictions.tolist()).issubset({'a', 'b'})

    artifacts = model.export_artifacts()
    assert artifacts['adapter'] == 'future_fusion_classifier'
    assert artifacts['train_duration_s'] > 0
    assert artifacts['training_history']['best_epoch'] >= 1
