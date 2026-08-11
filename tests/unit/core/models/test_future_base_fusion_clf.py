import math

import pytest
import torch
import torch.nn as nn

from fedot_ind.core.models.future.fusion.bottleneck_encoder import (
    BottleneckRepresentationEncoder,
)
from fedot_ind.core.models.future.future_clf import (
    ConfigurableMultimodalFusionClassifier,
)
from fedot_ind.core.models.future.enums import FusionInputsParam
from fedot_ind.core.models.future.mapping import FUSION_REGISTRY, FusionMethod
from fedot_ind.core.models.future.rules import (
    require_initialized_model_parts,
    require_resolved_modalities,
    validate_choice,
    validate_context_modalities_for_raw_centered,
    validate_divisible,
    validate_embeddings_count,
    validate_encoder_registry_has_modalities,
    validate_multimodal_bundle_input,
    validate_positive_int,
    validate_stacked_embeddings_shape,
)
from fedot_ind.core.models.nn.network_impl.mapping import EncoderShapeArg
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.mapping import MODALITY_CAPABILITIES


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


def test_modality_registry_contains_mvp_encoder_presets():
    assert {
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    }.issubset(set(MODALITY_CAPABILITIES))
    assert (
        MODALITY_CAPABILITIES[MultimodalModality.stats].encoder_preset.shape_arg_name
        is EncoderShapeArg.in_features
    )
    for modality in (
        MultimodalModality.raw,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    ):
        assert (
            MODALITY_CAPABILITIES[modality].encoder_preset.shape_arg_name
            is EncoderShapeArg.in_channels
        )


def test_fusion_registry_contains_mvp_methods():
    mvp_methods = {
        FusionMethod.concat,
        FusionMethod.gated,
        FusionMethod.raw_centered_residual,
        FusionMethod.film,
    }
    assert mvp_methods.issubset(set(FUSION_REGISTRY.keys()))
    for method in mvp_methods:
        assert FUSION_REGISTRY[method].is_extended is False


def test_fusion_registry_contains_extended_bottleneck_methods():
    bottleneck_methods = {
        FusionMethod.ordinary_bottleneck,
        FusionMethod.raw_residual_bottleneck,
        FusionMethod.context_only_residual_bottleneck,
    }
    assert bottleneck_methods.issubset(set(FUSION_REGISTRY.keys()))
    for method in bottleneck_methods:
        assert FUSION_REGISTRY[method].is_extended is True

    assert FUSION_REGISTRY[FusionMethod.ordinary_bottleneck].requires_raw is False
    assert FUSION_REGISTRY[FusionMethod.raw_residual_bottleneck].requires_raw is True
    assert (
        FUSION_REGISTRY[FusionMethod.context_only_residual_bottleneck].requires_raw
        is True
    )
    assert (
        FUSION_REGISTRY[FusionMethod.ordinary_bottleneck].inputs_param
        is FusionInputsParam.n_inputs
    )
    assert (
        FUSION_REGISTRY[FusionMethod.raw_residual_bottleneck].inputs_param
        is FusionInputsParam.n_inputs
    )
    assert (
        FUSION_REGISTRY[FusionMethod.context_only_residual_bottleneck].inputs_param
        is FusionInputsParam.n_context_inputs
    )
    assert (
        FUSION_REGISTRY[FusionMethod.raw_residual_bottleneck].resolve_input_count(4)
        == 4
    )
    assert (
        FUSION_REGISTRY[
            FusionMethod.context_only_residual_bottleneck
        ].resolve_input_count(4)
        == 3
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


@pytest.mark.parametrize(
    "fusion_method,modalities",
    [
        (
            "ordinary_bottleneck",
            (
                MultimodalModality.raw,
                MultimodalModality.stats,
                MultimodalModality.gaf,
            ),
        ),
        (
            "raw_residual_bottleneck",
            (
                MultimodalModality.raw,
                MultimodalModality.stats,
                MultimodalModality.stft,
            ),
        ),
        (
            "context_only_residual_bottleneck",
            (
                MultimodalModality.raw,
                MultimodalModality.stats,
                MultimodalModality.gaf,
            ),
        ),
    ],
)
def test_bottleneck_fusion_forward_shape_cpu(fusion_method, modalities):
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=modalities,
        num_classes=3,
        fusion_method=fusion_method,
        d_model=16,
        fusion_kwargs={"num_heads": 4, "num_latents": 2, "num_layers": 1},
    )
    model.build(bundle)
    logits = model(bundle)
    assert logits.shape == (bundle.n_samples, 3)


def test_ordinary_bottleneck_returns_attention_diagnostics():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
        ),
        num_classes=2,
        fusion_method="ordinary_bottleneck",
        d_model=16,
        fusion_kwargs={
            "num_heads": 4,
            "num_latents": 2,
            "num_layers": 1,
            "pooling": "mean",
        },
    )
    model.build(bundle)
    aux = model(bundle, return_aux=True)
    assert aux.attention_summary is not None
    assert "cross_attn_mean" in aux.attention_summary
    assert "self_attn_mean" in aux.attention_summary
    assert aux.pooling == "mean"
    assert aux.num_latents == 2
    assert aux.num_heads == 4
    assert aux.num_layers == 1
    assert aux.num_parameters is not None
    assert aux.num_parameters["fusion"] > 0
    assert aux.alpha is None


@pytest.mark.parametrize(
    "fusion_method",
    ["raw_residual_bottleneck", "context_only_residual_bottleneck"],
)
def test_residual_bottleneck_returns_alpha(fusion_method):
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.stft,
        ),
        num_classes=2,
        fusion_method=fusion_method,
        d_model=16,
        raw_modality=MultimodalModality.raw,
        fusion_kwargs={"num_heads": 4, "num_latents": 2},
    )
    model.build(bundle)
    aux = model(bundle, return_aux=True)
    assert aux.alpha is not None
    assert aux.alpha.shape[0] == bundle.n_samples
    assert aux.alpha_stats is not None
    assert aux.delta is not None
    assert aux.delta.shape == (bundle.n_samples, 16)
    assert aux.h_context is not None
    assert aux.h_context.shape == (bundle.n_samples, 16)
    assert aux.h_raw is not None
    assert aux.h_raw.shape == (bundle.n_samples, 16)
    assert aux.attention_summary is not None
    assert aux.pooling == "mean"
    assert aux.alpha_stats["mean"] > 0.0
    assert aux.delta.detach().norm().item() > 0.0


def test_residual_bottleneck_requires_raw_in_modalities():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.stats: torch.randn(4, 12),
            MultimodalModality.gaf: torch.randn(4, 1, 16, 16),
        },
    )
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(MultimodalModality.stats, MultimodalModality.gaf),
        num_classes=2,
        fusion_method="raw_residual_bottleneck",
        d_model=16,
        raw_modality=MultimodalModality.raw,
        fusion_kwargs={"num_heads": 4},
    )
    with pytest.raises(ValueError, match="requires raw modality"):
        model.build(bundle)


def test_bottleneck_missing_modality_in_bundle_raises():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(4, 1, 32),
            MultimodalModality.gaf: torch.randn(4, 1, 16, 16),
        },
    )
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
        ),
        num_classes=2,
        fusion_method="ordinary_bottleneck",
        d_model=16,
        fusion_kwargs={"num_heads": 4, "num_latents": 2},
    )
    model.build(_make_bundle())
    with pytest.raises(ValueError, match="does not contain required modalities"):
        model(bundle)


def test_bottleneck_invalid_pooling_via_build_raises():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
        ),
        num_classes=2,
        fusion_method="ordinary_bottleneck",
        d_model=16,
        fusion_kwargs={"num_heads": 4, "pooling": "max"},
    )
    with pytest.raises(ValueError, match="Unknown pooling"):
        model.build(bundle)


def test_bottleneck_invalid_num_heads_via_build_raises():
    bundle = _make_bundle()
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
        ),
        num_classes=2,
        fusion_method="ordinary_bottleneck",
        d_model=16,
        fusion_kwargs={"num_heads": 5, "num_latents": 2},
    )
    with pytest.raises(ValueError, match="divisible by num_heads"):
        model.build(bundle)


@pytest.mark.parametrize(
    "fusion_method",
    [
        "ordinary_bottleneck",
        "raw_residual_bottleneck",
        "context_only_residual_bottleneck",
    ],
)
def test_bottleneck_smoke_train_synthetic_cpu(fusion_method):
    torch.manual_seed(0)
    batch_size = 8
    num_classes = 3
    epochs = 2
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(batch_size, 1, 32),
            MultimodalModality.stats: torch.randn(batch_size, 12),
            MultimodalModality.gaf: torch.randn(batch_size, 1, 16, 16),
        },
        target=torch.randint(0, num_classes, (batch_size,)),
    )
    model = ConfigurableMultimodalFusionClassifier(
        modalities=(
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
        ),
        num_classes=num_classes,
        fusion_method=fusion_method,
        d_model=16,
        raw_modality=MultimodalModality.raw,
        fusion_kwargs={"num_heads": 4, "num_latents": 2, "num_layers": 1},
    )
    model.build(bundle)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    last_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(bundle)
        loss = criterion(logits, bundle.target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())

    assert logits.shape == (batch_size, num_classes)
    assert last_loss is not None
    assert math.isfinite(last_loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


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


def test_future_rules_validate_common_failure_surfaces():
    with pytest.raises(ValueError, match="num_classes"):
        validate_positive_int(name="num_classes", value=0)

    with pytest.raises(ValueError, match="Unknown pooling"):
        validate_choice(
            name="pooling",
            value="max",
            allowed=("mean", "cls", "concat"),
        )

    with pytest.raises(ValueError, match="divisible by num_heads"):
        validate_divisible(
            dividend_name="d_model",
            dividend=16,
            divisor_name="num_heads",
            divisor=5,
        )

    with pytest.raises(TypeError, match="MultimodalDataBundle"):
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


@pytest.mark.parametrize("pooling", ["mean", "cls", "concat"])
def test_bottleneck_representation_encoder_pooling_modes(pooling):
    encoder = BottleneckRepresentationEncoder(
        n_modalities=3,
        d_model=16,
        num_latents=2,
        num_heads=4,
        pooling=pooling,
    )
    embeddings = (torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 16))
    h_final = encoder(*embeddings)
    assert h_final.shape == (4, 16)


def test_bottleneck_representation_encoder_rejects_incompatible_heads():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        BottleneckRepresentationEncoder(
            n_modalities=2,
            d_model=16,
            num_heads=5,
        )
