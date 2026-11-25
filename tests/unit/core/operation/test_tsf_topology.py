import numpy as np
import pytest

from fedot_ind.core.operation.transformation.representation.topological.window_analysis import (
    cknn_graph, manifold_topology_divergence, rolling_landscape, subwindow_features,
)


def test_cknn_symmetry_nestedness_and_permutation():
    x = np.array([[0., 0.], [1., 0.], [3., 0.], [3., 0.]])
    a, b = [cknn_graph(x, k=1, delta=t) for t in (.5, 2.)]
    assert not np.any(a & ~b)
    assert np.array_equal(b, b.T) and not np.any(np.diag(b))
    order = [2, 1, 3, 0]
    np.testing.assert_array_equal(cknn_graph(x[order], k=1, delta=2.), b[np.ix_(order, order)])
    assert not b[2].any()


def test_window_aggregation_and_landscapes():
    pytest.importorskip("ripser")
    signal = np.sin(np.arange(22.) / 2)
    args = dict(window=10, subwindow=6, stride=2)
    aggregate = subwindow_features(signal, **args)
    independent = np.stack([subwindow_features(signal[i:i + 10], **args)[0] for i in range(0, 13, 2)])
    np.testing.assert_allclose(aggregate, independent)
    points = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    one = rolling_landscape(points, window=4)
    two = rolling_landscape(points * 2, window=4)
    np.testing.assert_array_equal(one.end_indices, [3])
    assert one.norms[0] > 0
    np.testing.assert_allclose(two.norms, one.norms * 2**1.5, rtol=1e-6)


def test_directed_cross_barcode_and_hook_cleanup():
    pytest.importorskip("ripser")
    x = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert manifold_topology_divergence(x, x, sample_size=4) == (0., 0.)
    directions = manifold_topology_divergence(x, x + 2, sample_size=4, seed=4)
    assert directions == manifold_topology_divergence(x, x + 2, sample_size=4, seed=4)
    torch = pytest.importorskip("torch")
    from fedot_ind.core.operation.transformation.representation.topological.activations import activation_topology

    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU())
    model[1].eval()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    result = activation_topology(model, torch.from_numpy(x).float(), ("0",), sample_size=4)
    assert result["0"]["rank"] <= 2 and model.training and not model[1].training
    assert not model[0]._forward_hooks
    assert all(torch.equal(v, snapshot[k]) for k, v in model.state_dict().items())
