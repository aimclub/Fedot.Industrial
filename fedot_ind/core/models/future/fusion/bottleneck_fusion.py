"""Compatibility import; tensor implementation lives in fedot_ind.neural."""
from fedot_ind.neural.fusion.bottleneck_fusion import (
    BottleneckFusionBase,
    OrdinaryBottleneckFusion,
    RawResidualBottleneckFusion,
    ContextOnlyResidualBottleneckFusion,
)

__all__ = [
    "BottleneckFusionBase",
    "OrdinaryBottleneckFusion",
    "RawResidualBottleneckFusion",
    "ContextOnlyResidualBottleneckFusion"]
