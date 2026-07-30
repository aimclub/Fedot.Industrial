"""Shared validation and normalization rules for NN model configs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.rules import normalize_modality


def build_encoder_config_map(
    entries: Sequence[tuple[MultimodalModality | str, Any]],
) -> dict[MultimodalModality, Any]:
    """Build a modality -> encoder config mapping with duplicate checks."""

    normalized: dict[MultimodalModality, Any] = {}
    for raw_modality, config in entries:
        modality = normalize_modality(raw_modality)
        if modality in normalized:
            raise ValueError(
                f"Duplicate modality definition in encoder config: {modality.value}."
            )
        normalized[modality] = config
    if not normalized:
        raise ValueError("Encoder configuration map must contain at least one modality.")
    return normalized
