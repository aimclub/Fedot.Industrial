"""Unit tests for shared torch backend IO helpers."""

from __future__ import annotations

import torch

from fedot_ind.core.operation.transformation.torch_backend.io import set_torch_seed


def test_set_torch_seed_none_is_noop():
    before = torch.rand(3)
    set_torch_seed(None)
    after = torch.rand(3)
    # Generators keep advancing; this only asserts the call does not raise.
    assert before.shape == after.shape == (3,)


def test_set_torch_seed_is_deterministic():
    set_torch_seed(123)
    first = torch.rand(4)
    set_torch_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)
