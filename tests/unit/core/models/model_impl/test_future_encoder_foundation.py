import pytest
import torch

from fedot_ind.core.models.nn.models_rules import build_encoder_config_map
from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
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


def test_future_encoder_stack_raises_for_missing_modality():
    stack = FutureEncoderStack(
        {
            MultimodalModality.raw: raw_encoder_config(in_channels=1, d_model=16),
            MultimodalModality.stats: stats_encoder_config(in_features=5, d_model=16),
        }
    )
    with pytest.raises(ValueError, match="Missing required modalities"):
        stack({MultimodalModality.raw: torch.randn(3, 1, 32)})


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
