import math

import pytest
import torch

from fedot_ind.core.metrics.loss.soft_dtw_torch import soft_dtw, soft_dtw_divergence


def test_value_matches_enumerating_every_path():
    a = torch.tensor([[[0.], [1.]]], dtype=torch.double)
    b = torch.tensor([[[1.], [1.]]], dtype=torch.double)
    gamma = .4
    # Diagonal, then the two alignments with two axial steps.
    expected = -gamma * math.log(sum(math.exp(-cost / gamma) for cost in (1., 2., 1.)))
    assert soft_dtw(a, b, gamma=gamma).item() == pytest.approx(expected)
    assert soft_dtw_divergence(a, a).item() == pytest.approx(0.)


def test_gradient_and_unequal_lengths():
    a = torch.tensor([[[.2, .7], [.4, .2]]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([[[.1, .4], [.1, .1], [.2, .1]]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: soft_dtw_divergence(x, y, gamma=.5), (a, b))
    with pytest.raises(ValueError, match="gamma"):
        soft_dtw(a, b, gamma=0.)
