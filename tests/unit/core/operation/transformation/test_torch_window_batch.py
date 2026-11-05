"""Window optimization must match the independent pre-optimization loop exactly."""
from types import SimpleNamespace

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
import torch

from fedot_ind.core.models.base_extractor import BaseExtractor
from fedot_ind.core.operation.transformation.representation.statistical.windows import statistical_window_plan
from fedot_ind.core.operation.transformation.torch_backend.statistical.quantile_extractor import TorchQuantileExtractor
from fedot_ind.core.operation.transformation.torch_backend.statistical.window_batch import window_statistics


class Reference(TorchQuantileExtractor):
    def apply_window_for_stat_feature_torch(self, ts_data, feature_generator, window_size=None):
        plan = statistical_window_plan(ts_data.shape[-1], self.window_size if window_size is None else window_size,
                                       self.stride, self.use_sliding_window)
        return torch.cat([self._as_feature_matrix(feature_generator(ts_data[..., s:s + plan.width]), ts_data)
                          for s in plan.starts], dim=-1)


@pytest.mark.parametrize('shape', [(1, 1, 100), (4, 3, 100), (4, 9, 28), (5, 28)])
@pytest.mark.parametrize('sliding,stride', [(True, 1), (True, 3), (False, 1)])
@pytest.mark.parametrize('global_features', [True, False])
def test_full_features_are_exact_including_dtype_order_and_input_ownership(shape, sliding, stride, global_features):
    values = (1000 + np.random.default_rng(42).normal(size=shape)).astype(np.float64)
    original = values.copy()
    params = dict(use_cache=False, window_size=10, use_sliding_window=sliding,
                  stride=stride, add_global_features=global_features)
    reference = Reference(params)
    optimized = TorchQuantileExtractor(params)
    expected = BaseExtractor._transform(reference, SimpleNamespace(features=values))
    rng_before = torch.random.get_rng_state().clone()
    actual = optimized._transform(SimpleNamespace(features=values))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    np.testing.assert_array_equal(values, original)


@given(st.integers(6, 120), st.integers(1, 5), st.sampled_from([5, 10, 25, 45]), st.integers(1, 3))
@settings(max_examples=20, deadline=None)
def test_bounded_window_blocks_preserve_reference_and_feature_order(length, channels, share, stride):
    values = torch.tensor(np.random.default_rng(length).normal(size=(channels, length)), dtype=torch.float32)
    plan = statistical_window_plan(length, share, stride, True)
    reference = Reference(dict(window_size=share, stride=stride, use_sliding_window=True))
    expected = reference.apply_window_for_stat_feature_torch(values, reference.get_statistical_features_torch)
    actual = window_statistics(values, plan, channels * plan.width * 2)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('kind', ['zero', 'constant', 'signed_zero', 'duplicates', 'large'])
def test_adversarial_values_keep_original_rounding(kind):
    values = torch.tensor(np.random.default_rng(13).normal(size=(3, 30)), dtype=torch.float32)
    if kind == 'zero':
        values[:] = 0
    elif kind == 'constant':
        values[:] = 7
    elif kind == 'signed_zero':
        values[:, ::2] = 0.
        values[:, 1::2] = -0.
    elif kind == 'duplicates':
        values = values.round()
    else:
        values *= 1e30
    reference = Reference(dict(window_size=25, stride=1, use_sliding_window=True))
    expected = reference.apply_window_for_stat_feature_torch(values, reference.get_statistical_features_torch)
    actual = window_statistics(values, statistical_window_plan(30, 25, 1, True), 10000)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[expected == 0]), torch.signbit(expected[expected == 0]))


def test_custom_feature_callbacks_keep_the_reference_path():
    instance = TorchQuantileExtractor(dict(window_size=25, stride=1, use_sliding_window=True))
    values = torch.arange(20., dtype=torch.float32).reshape(1, -1)
    calls = []

    def custom(x):
        calls.append(x.shape)
        return [x.sum(-1)]
    result = instance.apply_window_for_stat_feature_torch(values, custom)
    assert len(calls) == 16 and result.shape == (1, 16)


def test_full_signed_zero_input_is_bitwise_equal_to_reference():
    values = np.zeros((4, 3, 30), dtype=np.float32)
    values[:, :, 1::2] = -0.
    params = dict(window_size=25, stride=1, use_sliding_window=True, use_cache=False)
    expected = BaseExtractor._transform(Reference(params), SimpleNamespace(features=values))
    actual = TorchQuantileExtractor(params)._transform(SimpleNamespace(features=values))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_sorted_window_temporary_respects_the_element_cap(monkeypatch):
    from fedot_ind.core.operation.transformation.torch_backend.statistical import window_batch
    compiled = window_batch._compiled_kernel()
    sizes = []

    def observe(block, width, step):
        sizes.append(block.unfold(-1, width, step).numel())
        return compiled(block, width, step)
    monkeypatch.setattr(window_batch, '_compiled_kernel', lambda: observe)
    values = torch.tensor(np.random.default_rng(14).normal(size=(3, 100)), dtype=torch.float32)
    actual = window_statistics(values, statistical_window_plan(100, 10, 1, True), 60)
    assert len(sizes) > 1 and max(sizes) <= 60 and actual.shape == (3, 91 * 9)


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_deferred_runtime_preserves_windowed_features_and_targets(task):
    from fedot_ind.integration.fedot.extensions.catalog import load_industrial_extension_catalog
    from fedot_ind.integration.fedot.extensions.factories import make_deferred_factory
    values = np.random.default_rng(36).normal(size=(4, 3, 28)).astype(np.float32)
    target = np.arange(4) % 2 if task == 'classification' else np.arange(4.)
    original, original_target = values.copy(), target.copy()
    params = dict(window_size=25, stride=3, use_sliding_window=True, use_cache=False, task_type=task)
    expected = BaseExtractor._transform(Reference(params), SimpleNamespace(features=values)).reshape(4, -1)
    runtime = make_deferred_factory(load_industrial_extension_catalog().operation('quantile_extractor_torch'))(params)
    runtime.fit(values, target)
    np.testing.assert_array_equal(runtime.transform(values), expected)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(target, original_target)
