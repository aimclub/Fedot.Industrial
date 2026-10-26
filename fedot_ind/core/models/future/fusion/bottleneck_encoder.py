"""Compatibility import; tensor implementation lives in fedot_ind.neural."""
from fedot_ind.neural.fusion.bottleneck_encoder import (
    BottleneckLatentBlock,
    BottleneckRepresentationEncoder,
)

__all__ = ["BottleneckLatentBlock", "BottleneckRepresentationEncoder"]
