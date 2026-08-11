"""Unit tests for FUTURE fusion aux merge helpers."""

from __future__ import annotations

import pytest
import torch

from fedot_ind.core.models.future.tools import (
    FusionAuxOutput,
    merge_fusion_aux_outputs,
)


def _make_aux(
    *,
    batch_size: int,
    num_classes: int = 3,
    embedding_dim: int = 4,
    include_optional: bool = True,
    include_embeddings: bool = False,
    attention_summary: dict[str, float] | None = None,
) -> FusionAuxOutput:
    aux = FusionAuxOutput(
        logits=torch.randn(batch_size, num_classes),
        h_final=torch.randn(batch_size, embedding_dim),
        active_modalities=["raw", "stats"],
        embedding_dim=embedding_dim,
        num_parameters={"total": 10},
        pooling="mean",
        num_latents=2,
        attention_summary=attention_summary,
    )
    if include_optional:
        aux.gates = torch.randn(batch_size, 2)
        aux.alpha = torch.ones(batch_size, embedding_dim)
        aux.gamma = torch.ones(batch_size, embedding_dim) * 2
        aux.beta = torch.ones(batch_size, embedding_dim) * 3
    if include_embeddings:
        aux.embeddings = {
            "raw": torch.randn(batch_size, embedding_dim),
            "stats": torch.randn(batch_size, embedding_dim),
        }
    return aux


def test_merge_fusion_aux_outputs_concatenates_batch_tensors():
    first = _make_aux(batch_size=2, include_optional=True, include_embeddings=True)
    second = _make_aux(batch_size=3, include_optional=True, include_embeddings=True)

    merged = merge_fusion_aux_outputs([first, second])

    assert merged.logits.shape == (5, 3)
    assert merged.h_final.shape == (5, 4)
    assert merged.gates is not None and merged.gates.shape == (5, 2)
    assert merged.embeddings is not None
    assert merged.embeddings["raw"].shape == (5, 4)
    assert merged.embeddings["stats"].shape == (5, 4)
    assert torch.equal(merged.logits[:2], first.logits)
    assert torch.equal(merged.logits[2:], second.logits)
    assert merged.num_parameters == {"total": 10}
    assert merged.pooling == "mean"
    assert merged.num_latents == 2


def test_merge_fusion_aux_outputs_keeps_missing_optional_tensors_none():
    first = _make_aux(batch_size=2, include_optional=False)
    second = _make_aux(batch_size=2, include_optional=False)

    merged = merge_fusion_aux_outputs([first, second])

    assert merged.gates is None
    assert merged.alpha is None
    assert merged.embeddings is None
    assert merged.alpha_stats is None
    assert merged.gamma_beta_summary is None


def test_merge_fusion_aux_outputs_recomputes_stats_and_averages_attention():
    first = _make_aux(
        batch_size=2,
        attention_summary={"mean_entropy": 1.0, "max_weight": 0.8},
    )
    second = _make_aux(
        batch_size=2,
        attention_summary={"mean_entropy": 3.0, "max_weight": 0.4},
    )

    merged = merge_fusion_aux_outputs([first, second])

    assert merged.attention_summary is not None
    assert merged.attention_summary["mean_entropy"] == pytest.approx(2.0)
    assert merged.attention_summary["max_weight"] == pytest.approx(0.6)
    assert merged.alpha_stats is not None
    assert merged.gamma_beta_summary is not None
    assert merged.gamma_beta_summary["gamma_mean"] == pytest.approx(2.0)
    assert merged.gamma_beta_summary["beta_mean"] == pytest.approx(3.0)


def test_merge_fusion_aux_outputs_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        merge_fusion_aux_outputs([])
