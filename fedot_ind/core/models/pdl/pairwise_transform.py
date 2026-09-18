"""Deprecated compatibility layer for legacy PDL preprocessing helpers.

The active PDL execution path uses ``pairwise_model`` and the strategy-based
helpers in ``pairwise_core``. This module remains only to keep old imports of
``PDCDataTransformer`` and ``SampleWeights`` working during the transition.
"""

from __future__ import annotations

import warnings

from fedot_ind.core.models.pdl.legacy_pairwise_transform import (
    PDCDataTransformer,
    SampleWeights,
)

warnings.warn(
    "fedot_ind.core.models.pdl.pairwise_transform is deprecated; "
    "import PDCDataTransformer and SampleWeights from "
    "fedot_ind.core.models.pdl.legacy_pairwise_transform instead.",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PDCDataTransformer", "SampleWeights"]
