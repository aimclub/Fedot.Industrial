"""Numerical tensor-block references, masks and ownership invariants."""
import pytest
import torch

from fedot_ind.neural.blocks.attention import MultiHeadAttention, ScaledDotProductAttention
from fedot_ind.neural.blocks.linear import LinLnDrop


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_attention_matches_direct_reference_with_masks_and_gradients(additive, residual):
    gen = torch.Generator().manual_seed(714)
    q = torch.randn(2, 2, 3, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 4, 5, generator=gen, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, 5, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    before = [x.detach().clone() for x in (q, k, v)]
    blocked = torch.zeros(3, 5, dtype=torch.bool)
    blocked[0, 1] = True
    padding = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
    mask = torch.zeros(3, 5, dtype=torch.float64).masked_fill(blocked, -torch.inf) if additive else blocked
    previous = torch.full((2, 2, 3, 5), 0.25, dtype=torch.float64) if residual else None
    layer = ScaledDotProductAttention(8, 2, res_attention=residual, lsa=True).double()
    result = layer(q, k, v, prev=previous, attn_mask=mask, key_padding_mask=padding)
    scores = q @ k * 0.5
    if previous is not None:
        scores = scores + previous
    scores = scores.masked_fill(blocked, -torch.inf).masked_fill(padding[:, None, None, :], -torch.inf)
    weights = scores.softmax(-1)
    expected = weights @ v
    torch.testing.assert_close(result[0], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result[1], weights, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result[1].sum(-1), torch.ones(2, 2, 3, dtype=torch.float64))
    result[0].square().sum().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v))
    assert layer.scale.grad is not None and torch.isfinite(layer.scale.grad)
    for tensor, original in zip((q, k, v), before):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_masked_cross_attention_ignores_padded_values_and_their_gradients():
    gen = torch.Generator().manual_seed(531)
    q = torch.randn(2, 3, 8, generator=gen, requires_grad=True)
    k = torch.randn(2, 5, 8, generator=gen, requires_grad=True)
    v = torch.randn(2, 5, 8, generator=gen, requires_grad=True)
    padding = torch.tensor([[False, False, False, True, True]] * 2)
    layer = MultiHeadAttention(8, 2).eval()
    actual, weights = layer(q, K=k, V=v, key_padding_mask=padding)
    altered_k, altered_v = k.detach().clone(), v.detach().clone()
    altered_k[:, 3:] += 10000
    altered_v[:, 3:] -= 10000
    expected, _ = layer(q, K=altered_k, V=altered_v, key_padding_mask=padding)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(weights[..., 3:]) == 0
    actual.square().sum().backward()
    assert torch.count_nonzero(k.grad[:, 3:]) == 0
    assert torch.count_nonzero(v.grad[:, 3:]) == 0
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)


@pytest.mark.parametrize("linear_first", [False, True])
def test_head_block_keeps_legacy_sequential_weight_names(linear_first):
    block = LinLnDrop(4, 3, ln=True, p=0.2, act=torch.nn.GELU(), lin_first=linear_first).eval()
    norm = torch.nn.LayerNorm(3 if linear_first else 4)
    linear = torch.nn.Linear(4, 3, bias=False)
    reference = (torch.nn.Sequential(linear, torch.nn.GELU(), norm, torch.nn.Dropout(0.2))
                 if linear_first else torch.nn.Sequential(norm, torch.nn.Dropout(0.2), linear, torch.nn.GELU()))
    reference.eval()
    assert list(reference.state_dict()) == list(block.state_dict())
    reference.load_state_dict(block.state_dict(), strict=True)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(block(x), reference(x), rtol=0, atol=0)


def test_legacy_head_block_remains_a_compatible_public_class():
    from fedot_ind.core.models.nn.network_modules.layers.linear_layers import LinLnDrop as LegacyHead
    legacy = LegacyHead(4, 3).eval()
    neutral = LinLnDrop(4, 3).eval()
    neutral.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(neutral(x), legacy(x), rtol=0, atol=0)
    assert LegacyHead.__module__ == "fedot_ind.core.models.nn.network_modules.layers.linear_layers"
