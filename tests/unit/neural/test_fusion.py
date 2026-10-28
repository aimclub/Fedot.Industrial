"""Fusion has no dependency on modality enums, bundles or forecasting."""
import pytest
import torch

from fedot_ind.neural.fusion import (
    ContextOnlyResidualBottleneckFusion, FiLMFusion, MultiConcatFusionMLP,
    MultiModalGatedFusion, OrdinaryBottleneckFusion, RawCenteredResidualFusion,
    RawResidualBottleneckFusion,
)


FACTORIES = [
    lambda: MultiConcatFusionMLP(2, d_model=8, hidden_dim=8, dropout=0),
    lambda: MultiModalGatedFusion(2, d_model=8, hidden_dim=8, dropout=0),
    lambda: FiLMFusion(1, d_model=8, context_hidden_dim=8, film_hidden_dim=8, dropout=0),
    lambda: RawCenteredResidualFusion(1, d_model=8, context_hidden_dim=8, delta_hidden_dim=8, dropout=0),
    lambda: OrdinaryBottleneckFusion(2, d_model=8, num_latents=2, num_heads=2, dropout=0),
    lambda: RawResidualBottleneckFusion(2, d_model=8, num_latents=2, num_heads=2, dropout=0),
    lambda: ContextOnlyResidualBottleneckFusion(1, d_model=8, num_latents=2, num_heads=2, dropout=0),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_renaming_streams_preserves_outputs_and_input_gradients(factory):
    module = factory().eval()
    x, y = torch.randn(3, 8, requires_grad=True), torch.randn(3, 8, requires_grad=True)
    original = module.fuse({"sensor": x, "weather": y}, ("sensor", "weather"), raw_modality="sensor")
    renamed = module.fuse({"temperature": y, "device": x}, ("device", "temperature"), raw_modality="device")
    torch.testing.assert_close(original, renamed, rtol=0, atol=0)
    original_grads = torch.autograd.grad(original.square().sum(), (x, y))
    renamed_grads = torch.autograd.grad(renamed.square().sum(), (x, y))
    for before, after in zip(original_grads, renamed_grads):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        assert torch.isfinite(after).all()


@pytest.mark.parametrize("factory", FACTORIES)
def test_batch_partition_and_strict_state_dict_restore(factory):
    module, restored = factory().eval(), factory().eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    x, y = torch.randn(5, 8), torch.randn(5, 8)
    expected = module.fuse({"primary": x, "context": y}, ("primary", "context"), raw_modality="primary")
    chunks = [restored.fuse({"primary": x[s], "context": y[s]}, ("primary", "context"), raw_modality="primary")
              for s in (slice(0, 2), slice(2, None))]
    torch.testing.assert_close(torch.cat(chunks), expected, rtol=1e-5, atol=1e-6)
    aux = module.fuse({"primary": x, "context": y}, ("primary", "context"), raw_modality="primary", return_aux=True)
    assert aux["h_final"].shape == (5, 8)
    torch.testing.assert_close(aux["h_final"], expected, rtol=1e-5, atol=1e-6)


def test_legacy_attention_summary_delegates_to_tensor_core():
    from fedot_ind.core.models.future.tools import summarize_bottleneck_attn
    from fedot_ind.neural.fusion.diagnostics import summarize_bottleneck_attn as summarize_core
    attention = [{"cross_attn": torch.ones(2, 2), "self_attn": torch.zeros(2, 2)}]
    assert summarize_bottleneck_attn(attention) == summarize_core(attention) == {
        "num_layers_with_attn": 1.0, "cross_attn_mean": 1.0, "cross_attn_std": 0.0,
        "self_attn_mean": 0.0, "self_attn_std": 0.0,
    }
