"""Shared enums for FUTURE model composition."""

from enum import Enum


class FusionMethod(str, Enum):
    """Supported FUTURE fusion methods."""

    concat = "concat"
    gated = "gated"
    film = "film"
    raw_centered_residual = "raw_centered_residual"
    ordinary_bottleneck = "ordinary_bottleneck"
    raw_residual_bottleneck = "raw_residual_bottleneck"
    context_only_residual_bottleneck = "context_only_residual_bottleneck"


class FusionInputsParam(str, Enum):
    """Constructor argument that receives the resolved fusion input count.

    ``n_inputs`` — all active modalities (count = N).
    ``n_context_inputs`` — all modalities except raw (count = N - 1).
    """

    n_inputs = "n_inputs"
    n_context_inputs = "n_context_inputs"

    @property
    def excludes_raw(self) -> bool:
        return self is FusionInputsParam.n_context_inputs
