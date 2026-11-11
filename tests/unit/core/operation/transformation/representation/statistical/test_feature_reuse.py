"""Value reuse must not become a cache of models or caller-owned storage."""

from contextvars import copy_context
from threading import Event, Thread
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given, strategies as st

from fedot_ind.core.operation.transformation.representation.statistical.feature_reuse import (
    FeatureReuseSession, feature_reuse_scope, reuse_statistical_features, run_with_feature_reuse,
)


def test_reuse_is_explicit_and_outputs_are_private():
    x = np.arange(6.).reshape(2, 3)
    compute = Mock(side_effect=lambda: x * 2)
    reuse_statistical_features(x, (), compute)
    reuse_statistical_features(x, (), compute)
    assert compute.call_count == 2
    with feature_reuse_scope() as session:
        first = reuse_statistical_features(x, (), compute)
        first[:] = -1
        second = reuse_statistical_features(x.copy(), (), compute)
        np.testing.assert_array_equal(second, x * 2)
        second[:] = -2
        np.testing.assert_array_equal(reuse_statistical_features(x, (), compute), x * 2)
        assert session.statistics.hits == 2
        assert session.statistics.misses == 1
    assert session.statistics.retained_bytes == 0


@given(st.lists(st.integers(-100, 100), min_size=1, max_size=30), st.integers(1, 10))
def test_content_not_shape_or_identity_controls_reuse(values, delta):
    x = np.array(values, dtype=float)
    compute = Mock(side_effect=lambda: x.copy())
    with feature_reuse_scope() as session:
        np.testing.assert_array_equal(reuse_statistical_features(x, (), compute), x)
        x[0] += delta
        np.testing.assert_array_equal(reuse_statistical_features(x, (), compute), x)
        np.testing.assert_array_equal(reuse_statistical_features(x.copy(), (), compute), x)
        assert session.statistics.misses == 2
        assert session.statistics.hits == 1


@pytest.mark.parametrize('change', ['dtype', 'shape', 'settings'])
def test_key_separates_dtype_shape_and_settings(change):
    x = np.arange(6, dtype=np.int64)
    other = x.view(np.float64) if change == 'dtype' else x.reshape(2, 3) if change == 'shape' else x
    with feature_reuse_scope() as session:
        reuse_statistical_features(x, (1,), lambda: x.copy())
        reuse_statistical_features(other, (2,) if change == 'settings' else (1,), lambda: other.copy())
        assert session.statistics.misses == 2


def test_nested_exception_and_late_context_cannot_leak_entries():
    x = np.ones(3)
    compute = Mock(return_value=x)
    with feature_reuse_scope() as outer:
        reuse_statistical_features(x, (), compute)
        with pytest.raises(RuntimeError), feature_reuse_scope() as inner:
            reuse_statistical_features(x, (), compute)
            late = copy_context()
            raise RuntimeError('expected')
        assert inner.statistics.retained_bytes == 0
        late.run(reuse_statistical_features, x, (), compute)
        assert inner.statistics.retained_bytes == 0
        reuse_statistical_features(x, (), compute)
        assert outer.statistics.hits == 1
    with feature_reuse_scope() as fresh:
        reuse_statistical_features(x, (), compute)
        assert fresh.statistics.hits == 0


@pytest.mark.parametrize('max_bytes,max_entries', [(64, 10), (1024, 1)])
def test_retention_is_bounded_and_evicted_values_are_recomputed(max_bytes, max_entries):
    x = np.ones(8)
    with feature_reuse_scope(max_bytes=max_bytes, max_entries=max_entries) as session:
        for key in (0, 1, 0):
            reuse_statistical_features(x, (key,), lambda: x.copy())
        assert session.statistics.misses == 3
        assert session.statistics.peak_bytes <= max_bytes


@pytest.mark.parametrize('values,budget', [(np.ones(100), 32), (np.array(['a'], dtype=object), 1024),
                                           (np.ones(2), 0)])
def test_unsupported_or_oversized_inputs_bypass(values, budget):
    compute = Mock(return_value=values)
    with feature_reuse_scope(max_bytes=budget) as session:
        for _ in range(2):
            reuse_statistical_features(values, (), compute)
        assert compute.call_count == 2
        assert session.statistics.bypasses == 2


def test_oversized_outputs_and_failures_are_never_saved():
    x = np.ones(1)
    with feature_reuse_scope(max_bytes=8) as session:
        compute = Mock(side_effect=[ValueError('expected'), np.ones(2), np.ones(2)])
        with pytest.raises(ValueError):
            reuse_statistical_features(x, (), compute)
        reuse_statistical_features(x, (), compute)
        reuse_statistical_features(x, (), compute)
        assert session.statistics.retained_bytes == 0
        assert compute.call_count == 3


def test_call_helper_never_reuses_a_previous_invocation():
    compute = Mock(return_value=np.ones(1))
    for _ in range(2):
        run_with_feature_reuse(reuse_statistical_features, np.ones(1), (), compute)
    assert compute.call_count == 2


@pytest.mark.parametrize('kind', ['fortran', 'view', 'subclass'])
def test_noncanonical_layouts_and_array_subclasses_bypass(kind):
    class CustomArray(np.ndarray):
        pass
    x = np.arange(12.).reshape(3, 4)
    other = np.asfortranarray(x) if kind == 'fortran' else x[:, ::-1] if kind == 'view' else x.view(CustomArray)
    compute = Mock(return_value=other)
    with feature_reuse_scope() as session:
        reuse_statistical_features(x, (), lambda: x.copy())
        for _ in range(2):
            np.testing.assert_array_equal(reuse_statistical_features(other, (), compute), other)
        assert session.statistics.bypasses == 2
        assert compute.call_count == 2


def test_close_does_not_wait_for_native_compute_or_retain_late_result():
    entered, release, closed = Event(), Event(), Event()
    session = FeatureReuseSession(1024, 2)

    def compute():
        entered.set()
        assert release.wait(5)
        return np.ones(2)

    worker = Thread(target=lambda: session.evaluate(np.ones(2), (), compute))
    worker.start()
    closer = Thread(target=lambda: (session.close(), closed.set()))
    try:
        assert entered.wait(5)
        closer.start()
        assert closed.wait(1), 'scope cleanup waited for native computation'
    finally:
        release.set()
        worker.join(5)
        closer.join(5)
    assert not worker.is_alive() and not closer.is_alive()
    assert session.statistics.retained_bytes == 0


def test_recursive_key_does_not_double_count_retained_bytes():
    x = np.ones(2)
    with feature_reuse_scope() as session:
        reuse_statistical_features(x, (), lambda: reuse_statistical_features(x, (), lambda: x.copy()))
        assert session.statistics.retained_bytes == x.nbytes


@pytest.mark.parametrize('budget,entries', [(-1, 1), (True, 1), (1., 1), (1, 0), (1, True)])
def test_invalid_limits_rejected(budget, entries):
    with pytest.raises(ValueError):
        FeatureReuseSession(budget, entries)
