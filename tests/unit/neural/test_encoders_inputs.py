import pytest
import torch

from fedot_ind.neural.encoders import EncoderStack, raw_encoder_config, stats_encoder_config


def configs():
    return {
        "sensor": raw_encoder_config(2, d_model=8, hidden_channels=(4,)),
        "weather": stats_encoder_config(5, d_model=8, hidden_dims=(4,)),
    }


@pytest.mark.parametrize("entries,match", [
    ([], "at least one encoder"),
    ([("sensor", raw_encoder_config(1)), ("sensor", raw_encoder_config(1))], "Duplicate stream"),
    ({"": raw_encoder_config(1)}, "non-empty"),
    ({"a.b": raw_encoder_config(1)}, "must not contain"),
    ({"a": raw_encoder_config(1, 8), "b": stats_encoder_config(2, 16)}, "same d_model"),
])
def test_stack_rejects_invalid_config_maps(entries, match):
    with pytest.raises(ValueError, match=match):
        EncoderStack(entries)


def test_stack_rejects_non_string_names():
    with pytest.raises(TypeError, match="must be strings"):
        EncoderStack({1: raw_encoder_config(1)})


@pytest.mark.parametrize("sensor,weather,error,match", [
    (torch.randn(2, 2, 12), None, ValueError, "required streams"),
    (torch.randn(2, 2), torch.randn(2, 5), ValueError, "expected rank"),
    (torch.randn(2, 3, 12), torch.randn(2, 5), ValueError, "axis 1 size"),
    (torch.randn(2, 2, 12), torch.randn(2, 4), ValueError, "axis 1 size"),
    (torch.randn(2, 2, 12), torch.randn(3, 5), ValueError, "same batch size"),
    (torch.randn(2, 2, 0), torch.randn(2, 5), ValueError, "non-empty feature axes"),
    ("invalid", torch.randn(2, 5), TypeError, "must be torch.Tensor"),
])
def test_stack_validates_all_inputs_before_running_encoders(sensor, weather, error, match):
    stack = EncoderStack(configs())
    before = {key: value.clone() for key, value in stack.state_dict().items()}
    tensors = {"sensor": sensor}
    if weather is not None:
        tensors["weather"] = weather
    with pytest.raises(error, match=match):
        stack(tensors)
    assert all(torch.equal(value, stack.state_dict()[key]) for key, value in before.items())


def test_missing_input_is_not_replaced_by_other_available_stream():
    stack = EncoderStack(configs())
    with pytest.raises(ValueError, match="weather"):
        stack({"sensor": torch.randn(2, 2, 12), "unrelated": torch.randn(2, 5)})
