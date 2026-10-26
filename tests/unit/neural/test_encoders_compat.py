import pytest
import torch
import io

from fedot_ind.core.models.nn.network_impl.future_encoder_adapter import FutureEncoderStack
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.neural.encoders import EncoderStack, raw_encoder_config, stats_encoder_config


@pytest.mark.parametrize("batch_size", [1, 4])
def test_domain_and_neutral_stacks_strict_state_dict_roundtrip(batch_size):
    configs = {
        "raw": raw_encoder_config(2, d_model=8, hidden_channels=(4,), dropout=0),
        "stats": stats_encoder_config(5, d_model=8, hidden_dims=(4,), dropout=0),
    }
    neutral = EncoderStack(configs).eval()
    domain = FutureEncoderStack({MultimodalModality(key): config for key, config in configs.items()}).eval()
    assert list(neutral.state_dict()) == list(domain.state_dict())
    assert all(key.startswith(("encoders.raw.", "encoders.stats.")) for key in domain.state_dict())
    domain.load_state_dict(neutral.state_dict(), strict=True)
    neutral.load_state_dict(domain.state_dict(), strict=True)
    tensors = {"raw": torch.randn(batch_size, 2, 12), "stats": torch.randn(batch_size, 5)}
    expected, neutral_aux = neutral(tensors, return_aux=True)
    actual, aux = domain({MultimodalModality(key): value for key, value in tensors.items()}, return_aux=True)
    assert all(type(key) is MultimodalModality for key in actual)
    assert all(torch.equal(expected[key.value], value) for key, value in actual.items())
    assert aux["shapes"] == neutral_aux["shapes"]
    assert aux["active_modalities"] == ["raw", "stats"]
    assert aux["num_parameters"]["per_modality"] == neutral_aux["num_parameters"]["per_stream"]
    assert "active_streams" not in aux


def test_domain_classes_and_rules_are_available_through_legacy_paths():
    from fedot_ind.core.models.nn import models_rules
    from fedot_ind.core.models.nn.network_impl import future_encoder_adapter
    from fedot_ind.core.multimodal import encoder_rules, encoders

    assert future_encoder_adapter is encoders
    assert models_rules.build_encoder_config_map is encoder_rules.build_encoder_config_map


def test_full_legacy_stack_checkpoint_restores_only_descriptive_metadata(monkeypatch):
    configs = {MultimodalModality.raw: raw_encoder_config(1, d_model=8, hidden_channels=(4,), dropout=0)}
    stack = FutureEncoderStack(configs).eval()
    tensors = {MultimodalModality.raw: torch.randn(2, 1, 12)}
    expected = stack(tensors)[MultimodalModality.raw].detach()
    expected_state = {key: value.clone() for key, value in stack.state_dict().items()}
    del stack.stream_names
    del stack._stream_configs
    monkeypatch.setattr(FutureEncoderStack, "__module__",
                        "fedot_ind.core.models.nn.network_impl.future_encoder_adapter")
    checkpoint = io.BytesIO()
    torch.save(stack, checkpoint)
    checkpoint.seek(0)
    rng = torch.get_rng_state().clone()
    restored = torch.load(checkpoint, weights_only=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert list(restored.state_dict()) == list(expected_state)
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected_state[key], rtol=0, atol=0)
    torch.testing.assert_close(restored(tensors)[MultimodalModality.raw], expected, rtol=0, atol=0)
