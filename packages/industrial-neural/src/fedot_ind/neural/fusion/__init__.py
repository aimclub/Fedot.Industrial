"""Fusion modules consume named embeddings, not task-specific data bundles."""
from .bottleneck_encoder import BottleneckLatentBlock, BottleneckRepresentationEncoder
from .bottleneck_fusion import (
    ContextOnlyResidualBottleneckFusion, OrdinaryBottleneckFusion, RawResidualBottleneckFusion,
)
from .concat_fusion import MultiConcatFusionMLP
from .film_fusion import FiLMFusion
from .gated_fusion import MultiModalGatedFusion
from .raw_centered_residual_fusion import RawCenteredResidualFusion

__all__ = [
    "BottleneckLatentBlock", "BottleneckRepresentationEncoder", "ContextOnlyResidualBottleneckFusion",
    "OrdinaryBottleneckFusion", "RawResidualBottleneckFusion", "MultiConcatFusionMLP", "FiLMFusion",
    "MultiModalGatedFusion", "RawCenteredResidualFusion",
]
