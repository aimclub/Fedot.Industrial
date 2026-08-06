"""Unit tests for FUTURE classifier trainer and multimodal batching."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fedot_ind.core.models.future import (
    ConfigurableMultimodalFusionClassifier,
    FutureClassifierTrainer,
    FutureTrainingConfig,
)
from fedot_ind.core.multimodal.batching import (
    make_bundle_dataloader,
    select_bundle_indices,
    split_bundle_by_fraction,
)
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


torch.set_num_threads(1)


def _make_supervised_bundle(
    batch_size: int = 16,
    num_classes: int = 3,
    seed: int = 0,
) -> MultimodalDataBundle:
    generator = torch.Generator().manual_seed(seed)
    return MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(
                batch_size, 1, 32, generator=generator
            ),
            MultimodalModality.stats: torch.randn(
                batch_size, 12, generator=generator
            ),
        },
        target=torch.randint(
            0, num_classes, (batch_size,), generator=generator
        ),
    )


_MODALITIES = (
    MultimodalModality.raw,
    MultimodalModality.stats,
)


def _make_loader(
    bundle: MultimodalDataBundle,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
):
    return make_bundle_dataloader(
        bundle,
        batch_size=batch_size,
        shuffle=shuffle,
        device="cpu",
        seed=seed,
        require_target=True,
    )


def test_select_bundle_indices_keeps_modalities_aligned():
    bundle = _make_supervised_bundle(batch_size=10)
    indices = [0, 3, 7]
    subset = select_bundle_indices(bundle, indices)

    assert subset.n_samples == 3
    assert torch.equal(subset.target, bundle.target[indices])
    for modality in bundle.modalities:
        assert torch.equal(
            subset.modalities[modality],
            bundle.modalities[modality][indices],
        )


def test_split_bundle_by_fraction_is_deterministic_with_seed():
    bundle = _make_supervised_bundle(batch_size=20)
    first_train, first_val = split_bundle_by_fraction(
        bundle, validation_fraction=0.25, seed=7
    )
    second_train, second_val = split_bundle_by_fraction(
        bundle, validation_fraction=0.25, seed=7
    )

    assert first_train.n_samples + first_val.n_samples == bundle.n_samples
    assert first_val.n_samples == 5
    assert torch.equal(first_train.target, second_train.target)
    assert torch.equal(first_val.target, second_val.target)


def test_bundle_dataloader_moves_batches_to_device():
    bundle = _make_supervised_bundle(batch_size=8)
    loader = make_bundle_dataloader(
        bundle,
        batch_size=3,
        shuffle=False,
        device="cpu",
        require_target=True,
    )
    batches = list(loader)
    assert len(batches) == 3
    assert [batch.n_samples for batch in batches] == [3, 3, 2]
    for batch in batches:
        assert batch.device.type == "cpu"
        assert batch.target is not None


@pytest.mark.parametrize(
    "fusion_method",
    ["concat", "ordinary_bottleneck", "raw_residual_bottleneck"],
)
def test_future_trainer_fit_predict_and_history(fusion_method, tmp_path: Path):
    num_classes = 3
    full_train = _make_supervised_bundle(batch_size=12, num_classes=num_classes)
    train_bundle, val_bundle = split_bundle_by_fraction(
        full_train, validation_fraction=0.25, seed=0
    )
    test_bundle = _make_supervised_bundle(
        batch_size=6, num_classes=num_classes, seed=1
    )

    fusion_kwargs = {}
    if "bottleneck" in fusion_method:
        fusion_kwargs = {"num_heads": 4, "num_latents": 2, "num_layers": 1}

    model = ConfigurableMultimodalFusionClassifier(
        modalities=_MODALITIES,
        num_classes=num_classes,
        fusion_method=fusion_method,
        d_model=16,
        fusion_kwargs=fusion_kwargs,
        raw_modality=MultimodalModality.raw,
    )
    trainer = FutureClassifierTrainer(
        model=model,
        config=FutureTrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            early_stopping_patience=5,
            device="cpu",
            seed=0,
        ),
    )

    history = trainer.fit(
        _make_loader(train_bundle, batch_size=4, shuffle=True, seed=0),
        _make_loader(val_bundle, batch_size=4, shuffle=False, seed=0),
        build_bundle=train_bundle,
    )

    assert len(history.train_loss) == 2
    assert len(history.validation_loss) == 2
    assert all(math.isfinite(loss) for loss in history.train_loss)
    assert all(
        loss is not None and math.isfinite(loss)
        for loss in history.validation_loss
    )
    assert history.train_duration_s > 0.0
    assert history.best_epoch >= 1
    assert history.num_parameters is not None and history.num_parameters > 0

    probabilities = trainer.predict_proba(test_bundle.without_target())
    predictions = trainer.predict(test_bundle.without_target())
    assert probabilities.shape == (test_bundle.n_samples, num_classes)
    assert predictions.shape == (test_bundle.n_samples,)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(test_bundle.n_samples))

    diagnostics = trainer.evaluate_diagnostics(test_bundle)
    assert diagnostics.logits.shape == (test_bundle.n_samples, num_classes)
    assert diagnostics.num_parameters is not None
    if fusion_method == "ordinary_bottleneck":
        assert diagnostics.attention_summary is not None
        assert diagnostics.pooling == "mean"

    checkpoint_path = tmp_path / "future_trainer.pt"
    trainer.save_checkpoint(checkpoint_path)
    restored = FutureClassifierTrainer.load_checkpoint(
        checkpoint_path, device="cpu"
    )
    restored_predictions = restored.predict(test_bundle.without_target())
    assert torch.equal(predictions, restored_predictions)


def test_future_trainer_restores_best_weights_with_early_stopping():
    train_bundle = _make_supervised_bundle(batch_size=16, num_classes=2, seed=2)
    val_bundle = _make_supervised_bundle(batch_size=8, num_classes=2, seed=3)

    model = ConfigurableMultimodalFusionClassifier(
        modalities=_MODALITIES,
        num_classes=2,
        fusion_method="concat",
        d_model=16,
    )
    trainer = FutureClassifierTrainer(
        model=model,
        config=FutureTrainingConfig(
            epochs=5,
            batch_size=4,
            learning_rate=1e-2,
            early_stopping_patience=1,
            device="cpu",
            seed=11,
        ),
    )
    history = trainer.fit(
        _make_loader(train_bundle, batch_size=4, shuffle=True, seed=11),
        _make_loader(val_bundle, batch_size=4, shuffle=False, seed=11),
        build_bundle=train_bundle,
    )

    assert 1 <= history.best_epoch <= len(history.train_loss)
    assert trainer._best_state_dict is not None
    current_state = trainer.model.state_dict()
    for key, value in trainer._best_state_dict.items():
        assert torch.equal(current_state[key], value)


def test_future_trainer_accepts_custom_optimizer_and_criterion():
    train_bundle = _make_supervised_bundle(batch_size=8, num_classes=2, seed=4)
    model = ConfigurableMultimodalFusionClassifier(
        modalities=_MODALITIES,
        num_classes=2,
        fusion_method="concat",
        d_model=16,
    )
    model.build(train_bundle)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    trainer = FutureClassifierTrainer(
        model=model,
        config=FutureTrainingConfig(epochs=1, batch_size=4, device="cpu", seed=4),
        optimizer=optimizer,
        criterion=criterion,
    )
    history = trainer.fit(
        _make_loader(train_bundle, batch_size=4, shuffle=False, seed=4),
        build_bundle=train_bundle,
    )
    assert history.train_loss
    assert trainer.optimizer is optimizer
    assert trainer.criterion is criterion


def test_future_trainer_requires_integer_targets():
    bundle = _make_supervised_bundle(batch_size=4)
    float_bundle = bundle.with_target(bundle.target.float())
    model = ConfigurableMultimodalFusionClassifier(
        modalities=_MODALITIES,
        num_classes=3,
        fusion_method="concat",
        d_model=16,
    )
    trainer = FutureClassifierTrainer(
        model=model,
        config=FutureTrainingConfig(epochs=1, batch_size=2, device="cpu"),
    )
    with pytest.raises(ValueError, match="integer class indices"):
        trainer.fit(
            _make_loader(float_bundle, batch_size=2),
            build_bundle=float_bundle,
        )
