"""Shared enums for FUTURE model composition."""

from enum import Enum


class FusionMethod(str, Enum):
    """Supported FUTURE fusion methods."""

    concat = "concat"
    gated = "gated"
    film = "film"
    raw_centered_residual = "raw_centered_residual"
