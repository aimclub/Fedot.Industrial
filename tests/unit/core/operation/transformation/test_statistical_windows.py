import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st

from fedot_ind.core.operation.transformation.representation.statistical.windows import statistical_window_plan
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor


@given(st.integers(1, 200), st.integers(1, 100), st.integers(1, 10), st.booleans())
def test_windows_are_nonempty_ordered_in_bounds_and_use_the_declared_step(length, share, stride, sliding):
    plan = statistical_window_plan(length, share, stride, sliding)
    assert plan.starts[0] == 0
    assert 1 <= plan.width <= length
    assert all(0 <= start <= length - plan.width for start in plan.starts)
    assert all(b - a == (stride if sliding else plan.width) for a, b in zip(plan.starts, plan.starts[1:]))


@pytest.mark.parametrize('sliding', [False, True])
@pytest.mark.parametrize('stride', [1, 3])
@pytest.mark.parametrize('length', [4, 24, 65])
def test_numpy_and_torch_share_window_order_and_never_window_the_sample_axis(sliding, stride, length):
    params = dict(window_size=25, stride=stride, use_sliding_window=sliding)
    values = np.arange(length, dtype=float)**2
    plan = statistical_window_plan(length, 25, stride, sliding)
    expected = np.array([np.mean(values[s:s + plan.width]) for s in plan.starts])
    numpy = QuantileExtractor(params)
    actual = numpy.apply_window_for_stat_feature(values, lambda x: np.array([np.mean(x)]), 25)
    np.testing.assert_allclose(actual, expected)
    tensor = TorchQuantileExtractor(params)
    batch = torch.tensor(np.stack([values, values + 100]))
    actual = tensor.apply_window_for_stat_feature_torch(batch, lambda x: [x.mean(dim=-1)], 25)
    np.testing.assert_allclose(actual.numpy(), np.stack([expected, expected + 100]))


@pytest.mark.parametrize('sliding', [False, True])
@pytest.mark.parametrize('global_features', [False, True])
def test_real_windowed_generators_keep_observations_and_finite_features(sliding, global_features):
    params = dict(window_size=25, stride=3, use_sliding_window=sliding, add_global_features=global_features)
    values = np.sin(np.arange(32.) * .3) + np.arange(32.) * .02
    numpy = QuantileExtractor(params).extract_stats_features(values, axis=0)
    tensor = TorchQuantileExtractor(params).generate_features_from_ts(torch.tensor(values))
    assert numpy.ndim == 1 and tensor.shape == (1, len(numpy))
    assert np.isfinite(numpy).all() and torch.isfinite(tensor).all()


@pytest.mark.parametrize('share,stride,sliding', [(0, 1, True), (101, 1, True), (25, 0, True), (True, 1, True),
                                                  (25, 1, 'true')])
def test_invalid_window_configuration_is_not_silently_clipped(share, stride, sliding):
    with pytest.raises(ValueError):
        statistical_window_plan(32, share, stride, sliding)
