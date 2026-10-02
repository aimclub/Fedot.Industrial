"""Round-trip laws shared by supervised and temporal execution plans."""

import json

import numpy as np
import pytest

from fedot_ind.integration.fedot import (
    DataProfile,
    DetectionExecutionPlan,
    ForecastingExecutionPlan,
    IntegrationTask,
    ModelExecutionPlan,
)
from fedot_ind.integration.fedot.parameter_codec import freeze_mapping, thaw_value


@pytest.mark.parametrize(
    "make_plan",
    [
        lambda params: ModelExecutionPlan(DataProfile.TENSOR, IntegrationTask.REGRESSION,
                                          "pdl_reg", params),
        lambda params: ForecastingExecutionPlan(DataProfile.TENSOR, "ar", 2, params),
        lambda params: DetectionExecutionPlan(DataProfile.TENSOR, "detector", parameters=params),
    ],
)
def test_parameter_containers_survive_freeze_thaw_without_aliasing(make_plan):
    source = {
        "nested": [{"tuple": (1, [2, 3]), "set": {(1, 2)}}, frozenset({4, 5})],
        "array": np.array([6, 7]),
    }
    plan = make_plan(source)
    source["nested"][0]["tuple"][1].append(99)
    source["array"][0] = 99

    runtime = plan.runtime_parameters()
    assert isinstance(runtime["nested"], list)
    assert isinstance(runtime["nested"][0]["tuple"], tuple)
    assert isinstance(runtime["nested"][0]["tuple"][1], list)
    assert runtime["nested"][0]["tuple"] == (1, [2, 3])
    assert isinstance(runtime["nested"][0]["set"], set)
    assert runtime["nested"][0]["set"] == {(1, 2)}
    assert isinstance(runtime["nested"][1], frozenset)
    np.testing.assert_array_equal(runtime["array"], [6, 7])

    runtime["nested"][0]["tuple"][1].append(88)
    runtime["array"][0] = 88
    restored = plan.runtime_parameters()
    assert restored["nested"][0]["tuple"] == (1, [2, 3])
    np.testing.assert_array_equal(restored["array"], [6, 7])
    assert make_plan(plan.parameters).runtime_parameters()["nested"] == restored["nested"]
    json.dumps(plan.to_dict())


@pytest.mark.parametrize("source", [
    np.array(7, dtype=np.int16),
    np.empty((0, 3), dtype=np.float32),
    np.arange(24, dtype=np.float64).reshape(4, 6)[::2, ::-2],
    np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
])
def test_array_parameters_preserve_dtype_shape_and_values_in_independent_copies(source):
    expected = source.copy()
    frozen = freeze_mapping({"weights": source})
    source[...] = np.zeros_like(source)

    snapshot = frozen["weights"]
    np.testing.assert_array_equal(snapshot, expected)
    assert snapshot.dtype == expected.dtype
    assert snapshot.shape == expected.shape
    with pytest.raises(ValueError):
        snapshot.setflags(write=True)

    first = thaw_value(frozen)["weights"]
    second = thaw_value(frozen)["weights"]
    assert first.flags.writeable
    assert not np.shares_memory(first, snapshot)
    assert not np.shares_memory(first, second)
    first[...] = np.ones_like(first)
    np.testing.assert_array_equal(second, expected)


def test_repeated_freezing_preserves_empty_container_types():
    source = {"list": [], "tuple": (), "set": set(), "frozenset": frozenset(), "dict": {}}

    frozen = freeze_mapping(freeze_mapping(source))
    restored = thaw_value(frozen)

    assert restored == source
    assert {key: type(value) for key, value in restored.items()} == {
        key: type(value) for key, value in source.items()
    }
    with pytest.raises(TypeError):
        frozen["dict"]["new"] = 1
