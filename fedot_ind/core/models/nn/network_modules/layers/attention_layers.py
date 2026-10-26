"""Compatibility index for tensor-only attention layers."""
from fedot_ind.neural.blocks.attention import MultiHeadAttention, ScaledDotProductAttention

__all__ = ["MultiHeadAttention", "ScaledDotProductAttention"]
