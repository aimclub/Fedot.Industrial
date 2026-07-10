import pytest
import torch

from fedot_ind.core.models.future.future_clf import (
    ConfigurableMultimodalFusionClassifier,
)
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


def _make_bundle(batch_size: int = 4) -> MultimodalDataBundle:
    return MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(batch_size, 1, 32),
            MultimodalModality.stats: torch.randn(batch_size, 12),
            MultimodalModality.gaf: torch.randn(batch_size, 1, 16, 16),
            MultimodalModality.stft: torch.randn(batch_size, 1, 17, 9),
        },
        target=torch.randint(0, 2, (batch_size,)),
    )


def test_concat_base_fusion_classifier_forward_shape():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=3,
        fusion_method="concat",
        d_model=24,
    )
    logits = model(bundle)
    assert logits.shape == (bundle.n_samples, 3)


def test_gated_base_fusion_classifier_returns_aux_gates():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="gated",
        d_model=16,
    )
    aux = model(bundle, return_aux=True)
    assert aux.logits.shape == (bundle.n_samples, 2)
    assert aux.gates is not None
    assert aux.gates.shape == (bundle.n_samples, 3)


def test_film_base_fusion_classifier_returns_gamma_beta():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="film",
        d_model=16,
        raw_modality=MultimodalModality.raw,
    )
    aux = model(bundle, return_aux=True)
    assert aux.gamma is not None
    assert aux.beta is not None
    assert aux.gamma.shape == (bundle.n_samples, 16)
    assert aux.beta.shape == (bundle.n_samples, 16)


def test_raw_centered_base_fusion_classifier_returns_alpha():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.stft),
        num_classes=2,
        fusion_method="raw_centered_residual",
        d_model=16,
        raw_modality=MultimodalModality.raw,
    )
    aux = model(bundle, return_aux=True)
    assert aux.alpha is not None
    assert aux.alpha.shape[0] == bundle.n_samples


def test_unknown_fusion_method_raises():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        ConfigurableMultimodalFusionClassifier(
            num_classes=2,
            fusion_method="unknown",
            d_model=16,
        )
