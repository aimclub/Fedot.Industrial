import pytest
import torch
import torch.nn as nn

from fedot_ind.core.models.future.future_clf import (
    ConfigurableMultimodalFusionClassifier,
)
from fedot_ind.core.models.future.mapping import FUSION_REGISTRY, FusionMethod
from fedot_ind.core.models.future.rules import (
    require_initialized_model_parts,
    require_resolved_modalities,
    resolve_modalities_from_bundle,
    validate_context_modalities_for_raw_centered,
    validate_embeddings_count,
    validate_encoder_registry_has_modalities,
    validate_multimodal_bundle_input,
    validate_positive_int,
    validate_stacked_embeddings_shape,
)
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_PRESET_BUILDERS
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


def test_encoder_registry_contains_mvp_modalities():
    assert {
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    }.issubset(set(ENCODER_PRESET_BUILDERS.keys()))


def test_fusion_registry_contains_mvp_methods():
    assert {
        FusionMethod.concat,
        FusionMethod.gated,
        FusionMethod.raw_centered_residual,
        FusionMethod.film,
    }.issubset(
        set(FUSION_REGISTRY.keys())
    )


@pytest.mark.parametrize(
    "fusion_method,modalities",
    [
        ("concat", (MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.gaf)),
        ("gated", (MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.gaf)),
        ("raw_centered_residual", (MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.stft)),
        ("film", (MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.gaf)),
    ],
)
def test_mvp_fusion_forward_shape_cpu(fusion_method, modalities):
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=modalities,
        num_classes=3,
        fusion_method=fusion_method,
        d_model=24,
    )
    model.build(bundle)
    logits = model(bundle)
    assert logits.shape == (bundle.n_samples, 3)


def test_fusion_classifier_requires_explicit_build_before_forward():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="concat",
        d_model=16,
    )

    with pytest.raises(ValueError, match="Model is not built"):
        model(bundle)


def test_fusion_classifier_can_keep_legacy_auto_build_mode():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="concat",
        d_model=16,
        auto_build_on_forward=True,
    )

    logits = model(bundle)

    assert logits.shape == (bundle.n_samples, 2)


def test_gated_base_fusion_classifier_returns_aux_gates():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="gated",
        d_model=16,
    )
    model.build(bundle)
    aux = model(bundle, return_aux=True)
    assert aux.logits.shape == (bundle.n_samples, 2)
    assert aux.gates is not None
    assert aux.gates.shape == (bundle.n_samples, 4)
    assert aux.active_modalities == ["raw", "stats", "gaf", "stft"]
    assert aux.embedding_dim == 16
    assert aux.num_parameters is not None
    assert aux.num_parameters["total"] > 0


def test_film_base_fusion_classifier_returns_gamma_beta():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        num_classes=2,
        fusion_method="film",
        d_model=16,
        raw_modality=MultimodalModality.raw,
    )
    model.build(bundle)
    aux = model(bundle, return_aux=True)
    assert aux.gamma is not None
    assert aux.beta is not None
    assert aux.gamma.shape == (bundle.n_samples, 16)
    assert aux.beta.shape == (bundle.n_samples, 16)
    assert aux.gamma_beta_summary is not None
    assert "gamma_l2_norm" in aux.gamma_beta_summary
    assert "beta_l2_norm" in aux.gamma_beta_summary


def test_raw_centered_base_fusion_classifier_returns_alpha():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.stft),
        num_classes=2,
        fusion_method="raw_centered_residual",
        d_model=16,
        raw_modality=MultimodalModality.raw,
    )
    model.build(bundle)
    aux = model(bundle, return_aux=True)
    assert aux.alpha is not None
    assert aux.alpha.shape[0] == bundle.n_samples
    assert aux.alpha_stats is not None
    assert "mean" in aux.alpha_stats
    assert "std" in aux.alpha_stats


def test_unknown_fusion_method_raises():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        ConfigurableMultimodalFusionClassifier(
            num_classes=2,
            fusion_method="unknown",
            d_model=16,
        )


def test_duplicate_modalities_raise():
    with pytest.raises(ValueError, match="Duplicate modalities"):
        ConfigurableMultimodalFusionClassifier(
            modalities=(MultimodalModality.raw, MultimodalModality.raw),
            num_classes=2,
            fusion_method="concat",
            d_model=16,
        )


def test_missing_modality_in_bundle_raises():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(4, 1, 32),
            MultimodalModality.gaf: torch.randn(4, 1, 16, 16),
        },
    )
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(MultimodalModality.raw, MultimodalModality.stats, MultimodalModality.gaf),
        num_classes=2,
        fusion_method="concat",
        d_model=16,
    )
    model.build(_make_bundle())
    with pytest.raises(ValueError, match="does not contain required modalities"):
        model(bundle)


def test_mtf_modality_supported_when_provided_explicitly():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(4, 1, 32),
            MultimodalModality.mtf: torch.randn(4, 1, 16, 16),
        },
    )
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(MultimodalModality.raw, MultimodalModality.mtf),
        num_classes=2,
        fusion_method="concat",
        d_model=16,
    )
    model.build(bundle)
    logits = model(bundle)
    assert logits.shape == (bundle.n_samples, 2)


def test_unknown_modality_raises():
    with pytest.raises(ValueError, match="Unsupported modality"):
        ConfigurableMultimodalFusionClassifier(
            modalities=(MultimodalModality.raw, "unknown_modality"),
            num_classes=2,
            fusion_method="concat",
            d_model=16,
        )


def test_future_rules_resolve_modalities_from_metadata_order():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(2, 1, 8),
            MultimodalModality.stats: torch.randn(2, 4),
        },
        metadata={"modalities": [MultimodalModality.stats, "raw"]},
    )

    assert resolve_modalities_from_bundle(bundle) == (
        MultimodalModality.stats,
        MultimodalModality.raw,
    )


def test_future_rules_validate_common_failure_surfaces():
    with pytest.raises(ValueError, match="num_classes"):
        validate_positive_int(name="num_classes", value=0)

    with pytest.raises(ValueError, match="MultimodalDataBundle"):
        validate_multimodal_bundle_input(object())

    with pytest.raises(ValueError, match="Unsupported modalities"):
        validate_encoder_registry_has_modalities(
            modalities=[MultimodalModality.raw],
            preset_registry={},
        )

    with pytest.raises(ValueError, match="requires raw modality"):
        validate_context_modalities_for_raw_centered(
            raw_modality=MultimodalModality.raw,
            modalities=[MultimodalModality.stats],
        )

    with pytest.raises(ValueError, match="at least one context"):
        validate_context_modalities_for_raw_centered(
            raw_modality=MultimodalModality.raw,
            modalities=[MultimodalModality.raw],
        )

    with pytest.raises(ValueError, match="not resolved"):
        require_resolved_modalities(None)

    with pytest.raises(ValueError, match="Expected 2 context embeddings"):
        validate_embeddings_count(
            embeddings=(torch.zeros(1, 4),),
            expected_count=2,
            label="context embeddings",
        )

    with pytest.raises(ValueError, match="n_inputs=2"):
        validate_stacked_embeddings_shape(
            torch.zeros(3, 1, 4),
            expected_n_inputs=2,
            expected_d_model=4,
        )

    with pytest.raises(ValueError, match="d_model=8"):
        validate_stacked_embeddings_shape(
            torch.zeros(3, 2, 4),
            expected_n_inputs=2,
            expected_d_model=8,
        )

    with pytest.raises(ValueError, match="encoders"):
        require_initialized_model_parts(
            encoders=None,
            fusion=nn.Identity(),
            modalities=[MultimodalModality.raw],
        )

    with pytest.raises(ValueError, match="Fusion module"):
        require_initialized_model_parts(
            encoders=nn.ModuleDict(),
            fusion=None,
            modalities=[MultimodalModality.raw],
        )
