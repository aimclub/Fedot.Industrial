"""Compatibility index for task-independent activation modules."""
from fedot_ind.neural.blocks.activation import (
    GEGLU, ReGLU, SwishBeta, SmeLU, Mish, get_activation_fn, pytorch_acts, pytorch_act_names,
)

__all__ = ["GEGLU", "ReGLU", "SwishBeta", "SmeLU", "Mish", "get_activation_fn", "pytorch_acts", "pytorch_act_names"]
