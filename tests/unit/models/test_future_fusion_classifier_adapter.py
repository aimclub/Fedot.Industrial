"""Smoke tests for FUTURE benchmark classification adapter."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.industrial.core import ModelSpec, RunStatus
from benchmark.industrial.errors import BenchmarkClassificationError
from benchmark.industrial.models.classification import (
    FutureFusionClassifierAdapter,
    build_classification_model,
)
from fedot_ind.core.architecture.preprocessing.label_mapping import ContiguousLabelEncoder


def _future_adapter(**param_overrides: object) -> FutureFusionClassifierAdapter:
    params: dict = {
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
    }
    params.update(param_overrides)
    return FutureFusionClassifierAdapter(name='FutureConcat', params=params)


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


def test_build_classification_model_unknown_adapter_lists_registry():
    with pytest.raises(BenchmarkClassificationError, match='Available adapters'):
        build_classification_model(
            ModelSpec(adapter_name='unknown_adapter', display_name='Unknown')
        )


def test_future_fusion_classifier_adapter_fit_predict_smoke():
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(20, 32))
    train_y = np.asarray(['a', 'b'] * 10, dtype=object)
    test_x = rng.normal(size=(6, 32))

    model = _future_adapter()
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


def test_future_fusion_classifier_remaps_gapped_integer_labels():
    rng = np.random.default_rng(1)
    train_x = rng.normal(size=(24, 32))
    train_y = np.asarray([0, 2, 5] * 8, dtype=np.int64)

    model = _future_adapter()
    model.fit(train_x, train_y)

    assert model.label_mapping_ == {'0': 0, '2': 1, '5': 2}
    assert model.label_encoder_.inverse_transform([0, 1, 2]) == [0, 2, 5]
    assert model.trainer_.model.num_classes == 3


def test_future_fusion_classifier_decode_rejects_oov_class_index():
    model = FutureFusionClassifierAdapter(name='FutureConcat')
    model.label_encoder_ = ContiguousLabelEncoder().fit(['a', 'b'])

    decoded = model._decode_predicted_labels(np.asarray([0, 1]))
    assert decoded.tolist() == ['a', 'b']

    with pytest.raises(
        BenchmarkClassificationError,
        match='out of vocabulary',
    ):
        model._decode_predicted_labels(np.asarray([0, 99]))


def test_future_fusion_derives_preparation_modalities_when_preparation_omitted():
    from fedot_ind.core.multimodal.enums import MultimodalModality

    rng = np.random.default_rng(2)
    train_x = rng.normal(size=(16, 32))
    train_y = np.asarray(['a', 'b'] * 8, dtype=object)

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
        },
    )
    model.fit(train_x, train_y)

    assert tuple(model.preparer_.config.modalities) == (MultimodalModality.raw,)


def test_future_fusion_derives_multiple_preparation_modalities_when_omitted():
    from fedot_ind.core.multimodal.enums import MultimodalModality

    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(16, 64))
    train_y = np.asarray(['a', 'b'] * 8, dtype=object)

    model = FutureFusionClassifierAdapter(
        name='FutureConcat',
        params={
            'fusion_method': 'concat',
            'd_model': 16,
            'modalities': ['raw', 'stats', 'stft'],
            'training': {
                'epochs': 1,
                'batch_size': 8,
                'device': 'cpu',
                'seed': 0,
            },
        },
    )
    model.fit(train_x, train_y)

    assert tuple(model.preparer_.config.modalities) == (
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.stft,
    )
    assert set(model.preparer_.config.modalities) == set(
        model.trainer_.model.modalities
    )


def test_future_fusion_defaults_preparation_to_raw_when_modalities_none():
    from fedot_ind.core.multimodal.enums import MultimodalModality

    rng = np.random.default_rng(3)
    train_x = rng.normal(size=(16, 32))
    train_y = np.asarray(['a', 'b'] * 8, dtype=object)

    model = FutureFusionClassifierAdapter(
        name='FutureConcat',
        params={
            'fusion_method': 'concat',
            'd_model': 16,
            'training': {
                'epochs': 1,
                'batch_size': 8,
                'device': 'cpu',
                'seed': 0,
            },
        },
    )
    model.fit(train_x, train_y)

    assert tuple(model.preparer_.config.modalities) == (MultimodalModality.raw,)
