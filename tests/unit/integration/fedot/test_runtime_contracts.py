from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import (
    DataProfile,
    IntegrationContractError,
    IntegrationErrorCode,
    PredictionBatch,
    RegressionRuntime,
    RuntimeState,
    create_regression_runtime,
)
from fedot_ind.integration.fedot.contracts import PreparedData


class RecordingRuntime(RegressionRuntime):
    profile = DataProfile.LEGACY

    def __init__(self, *, fail_phase=None):
        super().__init__()
        self.fail_phase = fail_phase
        self.fitted_target = None
        self.closed = 0

    def _fit(self, data: PreparedData, target: np.ndarray) -> None:
        if self.fail_phase == "fit":
            raise RuntimeError("fit failed")
        self.fitted_target = target

    def _predict(self, data: PreparedData) -> np.ndarray:
        if self.fail_phase == "predict":
            raise RuntimeError("predict failed")
        return data.values[:, 0] * 2

    def _close(self) -> None:
        self.closed += 1


def test_runtime_lifecycle_and_prediction_coordinates_are_explicit():
    runtime = RecordingRuntime()
    train = pd.DataFrame({"signal": [1.0, 2.0]}, index=["a", "b"])
    target = np.array([3.0, 5.0])

    assert runtime.snapshot.state is RuntimeState.CREATED
    assert runtime.fit(train, target) is runtime
    assert runtime.snapshot.state is RuntimeState.FITTED
    result = runtime.predict(pd.DataFrame({"signal": [4.0]}, index=["future"]))

    assert isinstance(result, PredictionBatch)
    assert result.values.tolist() == [8.0]
    assert result.idx.tolist() == ["future"]
    assert not runtime.fitted_target.flags.writeable
    runtime.close()
    runtime.close()
    assert runtime.snapshot.state is RuntimeState.CLOSED and runtime.closed == 1


@pytest.mark.parametrize("action", ["predict-before-fit", "fit-twice", "fit-after-close"])
def test_invalid_state_transitions_are_structured(action):
    runtime = RecordingRuntime()
    data = np.array([[1.0], [2.0]])
    target = np.array([1.0, 2.0])
    if action == "fit-twice":
        runtime.fit(data, target)
    elif action == "fit-after-close":
        runtime.close()

    with pytest.raises(IntegrationContractError) as error:
        runtime.predict(data) if action == "predict-before-fit" else runtime.fit(data, target)

    assert error.value.code is IntegrationErrorCode.INVALID_STATE


def test_target_shape_and_length_are_validated_before_effects():
    runtime = RecordingRuntime()
    with pytest.raises(IntegrationContractError) as rank_error:
        runtime.fit(np.ones((2, 1)), np.ones((2, 2)))
    assert rank_error.value.code is IntegrationErrorCode.INVALID_DATA
    with pytest.raises(IntegrationContractError) as length_error:
        runtime.fit(np.ones((2, 1)), np.ones(1))
    assert length_error.value.code is IntegrationErrorCode.LENGTH_MISMATCH
    assert runtime.snapshot.state is RuntimeState.CREATED


@pytest.mark.parametrize("phase", ["fit", "predict"])
def test_runtime_failures_keep_phase_and_original_cause(phase):
    runtime = RecordingRuntime(fail_phase=phase)
    data = np.ones((2, 1))
    if phase == "predict":
        runtime.fail_phase = None
        runtime.fit(data, np.ones(2))
        runtime.fail_phase = "predict"
    with pytest.raises(IntegrationContractError) as error:
        runtime.fit(data, np.ones(2)) if phase == "fit" else runtime.predict(data)
    assert error.value.code is IntegrationErrorCode.RUNTIME_FAILURE
    assert error.value.context["phase"] == phase
    assert isinstance(error.value.cause, RuntimeError)


def test_prediction_schema_is_checked_before_runtime_call():
    runtime = RecordingRuntime().fit(pd.DataFrame({"a": [1.0], "b": [2.0]}), [1.0])
    with pytest.raises(IntegrationContractError) as error:
        runtime.predict(pd.DataFrame({"b": [2.0], "a": [1.0]}))
    assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH


def test_factory_imports_only_the_selected_profile(monkeypatch):
    from fedot_ind.integration.fedot import runtime as runtime_module

    imported = []

    class Legacy(RecordingRuntime):
        pass

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(LegacyRegressionRuntime=Legacy)

    monkeypatch.setattr(runtime_module, "import_module", fake_import)
    assert isinstance(create_regression_runtime("legacy"), Legacy)
    assert imported == ["fedot_ind.integration.fedot.legacy"]


def test_factory_rejects_unknown_profile_without_importing_runtime(monkeypatch):
    from fedot_ind.integration.fedot import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "import_module", lambda name: pytest.fail(name))
    with pytest.raises(IntegrationContractError) as error:
        create_regression_runtime("automatic")
    assert error.value.code is IntegrationErrorCode.UNKNOWN_PROFILE
