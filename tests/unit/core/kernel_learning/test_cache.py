import numpy as np

from fedot_ind.core.kernel_learning import (
    InMemoryKernelCache,
    KernelBundle,
    KernelCacheKey,
    fingerprint_array,
    fingerprint_mapping,
)


def test_in_memory_kernel_cache_round_trips_bundle_by_stable_key():
    key = KernelCacheKey(
        namespace="test",
        generator_name="identity",
        kernel_policy_hash=fingerprint_mapping({"kernel": "linear"}),
        data_fingerprint=fingerprint_array(np.eye(2)),
    )
    bundle = KernelBundle(name="identity", train_kernel=np.eye(2))
    cache = InMemoryKernelCache()

    assert cache.get(key) is None
    cache.put(key, bundle)

    assert cache.get(key) is bundle
    assert cache.size == 1


def test_fingerprint_array_is_stable_for_object_labels():
    left = fingerprint_array(np.array(["a", "b", "a"], dtype=object))
    right = fingerprint_array(np.array(["a", "b", "a"], dtype=object))

    assert left == right
