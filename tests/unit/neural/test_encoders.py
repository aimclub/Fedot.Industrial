import importlib
import subprocess
import sys
from enum import Enum

import pytest
import torch

from fedot_ind.neural.encoders import (
    EncoderStack,
    build_encoder,
    gaf_encoder_config,
    raw_encoder_config,
    stats_encoder_config,
    stft_encoder_config,
)


def small_configs():
    return {
        "sensor": raw_encoder_config(2, d_model=8, hidden_channels=(4,), dropout=0),
        "weather": stats_encoder_config(5, d_model=8, hidden_dims=(4,), dropout=0),
    }


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_named_stack_forward_backward_and_aux(batch_size):
    stack = EncoderStack(small_configs())
    tensors = {
        "sensor": torch.randn(batch_size, 2, 12, requires_grad=True),
        "weather": torch.randn(batch_size, 5, requires_grad=True),
    }
    embeddings, aux = stack(tensors, return_aux=True)
    assert list(embeddings) == ["sensor", "weather"]
    assert all(value.shape == (batch_size, 8) for value in embeddings.values())
    sum(value.square().sum() for value in embeddings.values()).backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all()
               for tensor in tensors.values())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in stack.parameters())
    assert aux["active_streams"] == ["sensor", "weather"]
    assert aux["shapes"]["input"] == {name: tuple(t.shape) for name, t in tensors.items()}
    assert aux["shapes"]["output"] == {name: (batch_size, 8) for name in tensors}
    counts = aux["num_parameters"]
    assert counts["total"] == sum(counts["per_stream"].values())


def test_stack_accepts_dynamic_batch_on_same_instance():
    stack = EncoderStack(small_configs())
    for batch in (2, 5):
        outputs = stack({"sensor": torch.randn(batch, 2, 12), "weather": torch.randn(batch, 5)})
        assert all(output.shape == (batch, 8) for output in outputs.values())


def test_string_subclass_keys_do_not_require_value_attribute():
    class StreamName(str):
        pass

    class Stream(str, Enum):
        sensor = "sensor"
        weather = "weather"

    configs = small_configs()
    stack = EncoderStack({Stream.sensor: configs["sensor"], Stream.weather: configs["weather"]})
    result = stack({StreamName("sensor"): torch.randn(2, 2, 12), "weather": torch.randn(2, 5)})
    assert list(result) == ["sensor", "weather"]
    assert all(type(key) is str for key in result)


@pytest.mark.parametrize("factory,shape", [
    (lambda: raw_encoder_config(2, d_model=8, hidden_channels=(4,)), (2, 2, 12)),
    (lambda: stats_encoder_config(5, d_model=8, hidden_dims=(4,)), (2, 5)),
    (lambda: gaf_encoder_config(2, d_model=8, hidden_channels=(4,)), (2, 2, 8, 8)),
    (lambda: stft_encoder_config(2, d_model=8, hidden_channels=(4,)), (2, 2, 9, 5)),
])
def test_family_encoder_cpu_backward(factory, shape):
    encoder = build_encoder(factory())
    tensor = torch.randn(*shape, requires_grad=True)
    output = encoder(tensor)
    output.square().sum().backward()
    assert output.shape == (shape[0], 8)
    assert torch.isfinite(tensor.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in encoder.parameters())


@pytest.mark.parametrize("module,names", [
    ("enums", ["EncoderFamily"]),
    ("config", ["EncoderConfig", "ConvBlockConfig"]),
    ("encoders", ["CNNEncoder", "MLPEncoder"]),
    ("builder", ["build_encoder"]),
    ("transformation_presets", [
        "raw_encoder_config", "stats_encoder_config", "gaf_encoder_config", "stft_encoder_config",
    ]),
])
def test_public_encoder_index_exports_canonical_implementations(module, names):
    public = importlib.import_module("fedot_ind.neural.encoders")
    implementation = importlib.import_module("fedot_ind.neural.encoders." + module)
    assert all(getattr(public, name) is getattr(implementation, name) for name in names)


def test_neutral_encoder_imports_without_domain_dependencies():
    script = '''
import importlib.abc
import sys
blocked = ("fedot", "golem", "FEDOT", "fedot_ind.core", "benchmark")
class RejectDomainImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise ImportError("Forbidden neural dependency: " + fullname)
sys.meta_path.insert(0, RejectDomainImports())
from fedot_ind.neural.encoders import EncoderStack, raw_encoder_config, stats_encoder_config
import torch
stack = EncoderStack({"sensor": raw_encoder_config(1, 8, (4,)),
                      "weather": stats_encoder_config(3, 8, (4,))})
result = stack({"sensor": torch.randn(2, 1, 8), "weather": torch.randn(2, 3)})
assert result["sensor"].shape == (2, 8)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
