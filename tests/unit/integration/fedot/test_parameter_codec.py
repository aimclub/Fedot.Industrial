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
