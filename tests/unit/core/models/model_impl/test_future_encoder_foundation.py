import pytest
import torch

from fedot_ind.core.models.nn.models_rules import (
    EncoderFamily,
    build_encoder_config_map,
    normalize_encoder_family,
    normalize_modality,
)
from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
from fedot_ind.core.models.nn.network_impl.encoders.config import (
    ConvBlockConfig,
    EncoderConfig,
)
from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
    gaf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)
from fedot_ind.core.models.nn.network_impl.future_encoder_adapter import (
    FutureEncoderStack,
    FutureMultimodalEncoderAdapter,
)
from fedot_ind.core.models.nn.network_impl.mapping import ENCODER_PRESET_BUILDERS
from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


@pytest.mark.parametrize(
    "config,input_shape",
    [
        (raw_encoder_config(in_channels=2, d_model=32), (4, 2, 64)),
        (stats_encoder_config(in_features=18, d_model=32), (4, 18)),
        (gaf_encoder_config(in_channels=2, d_model=32), (4, 2, 16, 16)),
        (stft_encoder_config(in_channels=2, d_model=32), (4, 2, 17, 9)),
    ],
)
def test_family_encoder_forward_shape(config, input_shape):
    encoder = build_encoder(config)
    output = encoder(torch.randn(*input_shape))
    assert output.shape == (input_shape[0], config.d_model)


def test_duplicate_modality_definition_in_config_map_raises():
    with pytest.raises(ValueError, match="Duplicate modality definition"):
        build_encoder_config_map(
            [
                (MultimodalModality.raw, raw_encoder_config(in_channels=1, d_model=16)),
                (MultimodalModality.raw, raw_encoder_config(in_channels=1, d_model=16)),
            ]
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        (EncoderFamily.cnn, EncoderFamily.cnn),
        ("mlp", EncoderFamily.mlp),
    ],
)
def test_normalize_encoder_family_accepts_enum_and_string(value, expected):
    assert normalize_encoder_family(value) is expected


def test_normalize_encoder_family_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported encoder family"):
        normalize_encoder_family("transformer")


@pytest.mark.parametrize(
    "value,expected",
    [
        (MultimodalModality.raw, MultimodalModality.raw),
        ("stats", MultimodalModality.stats),
    ],
)
def test_normalize_modality_accepts_enum_and_string(value, expected):
    assert normalize_modality(value) is expected


def test_normalize_modality_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported modality"):
        normalize_modality("spectral")


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"out_channels": 0}, "out_channels"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"stride": 0}, "stride"),
        ({"pool_kernel_size": 0}, "pool_kernel_size"),
        ({"pool_stride": 0}, "pool_stride"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_conv_block_config_rejects_invalid_values(updates, match):
    params = {"out_channels": 4}
    params.update(updates)

    with pytest.raises(ValueError, match=match):
        ConvBlockConfig(**params)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"d_model": 0}, "d_model"),
        ({"dropout": -0.1}, "dropout"),
        ({"input_rank": 2}, "CNN encoder input_rank"),
        ({"in_channels": None}, "positive in_channels"),
        ({"conv_blocks": ()}, "at least one conv block"),
        ({"hidden_dims": (8,)}, "must not define hidden_dims"),
        ({"in_features": 4}, "must not define in_features"),
    ],
)
def test_cnn_encoder_config_rejects_invalid_contracts(updates, match):
    params = {
        "family": EncoderFamily.cnn,
        "d_model": 16,
        "input_rank": 3,
        "in_channels": 1,
        "conv_blocks": (ConvBlockConfig(out_channels=4),),
    }
    params.update(updates)

    with pytest.raises(ValueError, match=match):
        EncoderConfig(**params)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"input_rank": 3}, "MLP encoder input_rank"),
        ({"in_features": None}, "positive in_features"),
        ({"hidden_dims": ()}, "non-empty hidden_dims"),
        ({"hidden_dims": (8, 0)}, "hidden_dims values"),
        ({"conv_blocks": (ConvBlockConfig(out_channels=4),)}, "must not define conv_blocks"),
        ({"in_channels": 1}, "must not define in_channels"),
    ],
)
def test_mlp_encoder_config_rejects_invalid_contracts(updates, match):
    params = {
        "family": EncoderFamily.mlp,
        "d_model": 16,
        "input_rank": 2,
        "in_features": 6,
        "hidden_dims": (8,),
    }
    params.update(updates)

    with pytest.raises(ValueError, match=match):
        EncoderConfig(**params)


def test_build_encoder_config_map_rejects_empty_entries():
    with pytest.raises(ValueError, match="at least one modality"):
        build_encoder_config_map([])


def test_future_encoder_stack_raises_for_missing_modality():
    stack = FutureEncoderStack(
        {
            MultimodalModality.raw: raw_encoder_config(in_channels=1, d_model=16),
            MultimodalModality.stats: stats_encoder_config(in_features=5, d_model=16),
        }
    )
    with pytest.raises(ValueError, match="Missing required modalities"):
        stack({MultimodalModality.raw: torch.randn(3, 1, 32)})


def test_future_encoder_stack_rejects_empty_and_mismatched_configs():
    with pytest.raises(ValueError, match="at least one encoder"):
        FutureEncoderStack({})

    with pytest.raises(ValueError, match="same d_model"):
        FutureEncoderStack(
            {
                MultimodalModality.raw: raw_encoder_config(in_channels=1, d_model=16),
                MultimodalModality.stats: stats_encoder_config(in_features=5, d_model=32),
            }
        )


def test_future_encoder_adapter_cpu_smoke_with_bundle():
    batch_size = 5
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(batch_size, 2, 64),
            MultimodalModality.stats: torch.randn(batch_size, 20),
            MultimodalModality.gaf: torch.randn(batch_size, 2, 16, 16),
            MultimodalModality.stft: torch.randn(batch_size, 2, 17, 9),
        },
        target=torch.randint(0, 2, (batch_size,)),
    )

    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 24})
    adapter.configure_from_bundle(
        bundle=bundle,
        modalities=[
            MultimodalModality.raw,
            MultimodalModality.stats,
            MultimodalModality.gaf,
            MultimodalModality.stft,
        ],
    )
    embeddings, aux = adapter.encode_bundle(bundle, return_aux=True)

    assert set(embeddings.keys()) == {
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    }
    for embedding in embeddings.values():
        assert embedding.shape == (batch_size, 24)

    assert aux["active_modalities"] == ["raw", "stats", "gaf", "stft"]
    assert aux["embedding_dim"] == 24
    assert aux["num_parameters"]["total"] > 0


def test_future_encoder_adapter_auto_configures_on_encode_bundle():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(3, 1, 16),
            MultimodalModality.stats: torch.randn(3, 6),
        }
    )
    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 12})

    embeddings = adapter.encode_bundle(bundle)

    assert set(embeddings) == {MultimodalModality.raw, MultimodalModality.stats}
    assert adapter.encoder_stack is not None


def test_future_encoder_adapter_requires_configuration_for_raw_modalities():
    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 16})

    with pytest.raises(ValueError, match="not configured"):
        adapter.encode_modalities({MultimodalModality.raw: torch.randn(2, 1, 16)})


def test_future_encoder_adapter_rejects_requested_missing_modality():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 16)}
    )
    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 16})

    with pytest.raises(ValueError, match="does not contain requested modality"):
        adapter.configure_from_bundle(
            bundle=bundle,
            modalities=[MultimodalModality.raw, MultimodalModality.stats],
        )


def test_future_encoder_adapter_rejects_registry_gap(monkeypatch):
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.mtf: torch.randn(2, 1, 8, 8)}
    )
    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 16})
    patched_registry = dict(ENCODER_PRESET_BUILDERS)
    patched_registry.pop(MultimodalModality.mtf)
    monkeypatch.setattr(
        "fedot_ind.core.models.nn.network_impl.future_encoder_adapter.ENCODER_PRESET_BUILDERS",
        patched_registry,
    )

    with pytest.raises(ValueError, match="Unsupported modalities"):
        adapter.configure_from_bundle(
            bundle=bundle,
            modalities=[MultimodalModality.mtf],
        )


def test_future_encoder_adapter_raises_on_duplicate_modalities():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(2, 1, 16),
            MultimodalModality.stats: torch.randn(2, 6),
        }
    )
    adapter = FutureMultimodalEncoderAdapter(params={"d_model": 16})
    with pytest.raises(ValueError, match="Duplicate modality"):
        adapter.configure_from_bundle(
            bundle=bundle,
            modalities=[MultimodalModality.raw, MultimodalModality.raw],
        )
