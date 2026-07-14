from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fedot_ind.api.services.prediction import PredictionService


class FakeConditionCheck:
    def __init__(self, *, is_fedot=True, is_pipeline=False, have_encoder=False):
        self.is_fedot = is_fedot
        self.is_pipeline = is_pipeline
        self.have_encoder = have_encoder

    def solver_have_target_encoder(self, encoder):
        return self.have_encoder and encoder is not None

    def solver_is_fedot_class(self, solver):
        return self.is_fedot

    def solver_is_pipeline_class(self, solver):
        return self.is_pipeline


def test_prediction_service_uses_solver_predict_for_labels():
    class FakeSolver:
        def __init__(self):
            self.calls = []

        def predict(self, data):
            self.calls.append(("predict", data))
            return np.array([1, 0])

        def predict_proba(self, data):
            self.calls.append(("predict_proba", data))
            return np.array([[0.2, 0.8]])

    solver = FakeSolver()
    manager = SimpleNamespace(solver=solver, condition_check=FakeConditionCheck())
    data = SimpleNamespace(task=SimpleNamespace(task_type=SimpleNamespace(value="classification")))

    result = PredictionService().predict_output(
        manager=manager,
        target_encoder=None,
        predict_data=data,
        predict_mode="labels",
    )

    np.testing.assert_array_equal(result, np.array([1, 0]))
    assert solver.calls == [("predict", data)]


def test_prediction_service_uses_solver_predict_proba_for_probabilities():
    class FakeSolver:
        def predict(self, data):
            raise AssertionError("labels path should not be used")

        def predict_proba(self, data):
            return np.array([[0.2, 0.8]])

    manager = SimpleNamespace(solver=FakeSolver(), condition_check=FakeConditionCheck())

    result = PredictionService().predict_output(
        manager=manager,
        target_encoder=None,
        predict_data=SimpleNamespace(task=SimpleNamespace(task_type=SimpleNamespace(value="classification"))),
        predict_mode="probs",
    )

    np.testing.assert_allclose(result, np.array([[0.2, 0.8]]))


def test_prediction_service_uses_pipeline_predict_signature():
    class FakePipeline:
        def __init__(self):
            self.calls = []

        def predict(self, data, mode):
            self.calls.append((data, mode))
            return np.array([1])

    pipeline = FakePipeline()
    manager = SimpleNamespace(
        solver=pipeline,
        condition_check=FakeConditionCheck(is_pipeline=True),
    )
    data = SimpleNamespace(task=SimpleNamespace(task_type=SimpleNamespace(value="classification")))

    result = PredictionService().predict_output(
        manager=manager,
        target_encoder=None,
        predict_data=data,
        predict_mode="labels",
    )

    np.testing.assert_array_equal(result, np.array([1]))
    assert pipeline.calls == [(data, "labels")]


def test_prediction_service_uses_custom_solver_path():
    class FakeCustomSolver:
        def predict(self, data):
            return {"custom": data}

    manager = SimpleNamespace(
        solver=FakeCustomSolver(),
        condition_check=FakeConditionCheck(is_fedot=False, is_pipeline=False),
    )
    data = SimpleNamespace(task=SimpleNamespace(task_type=SimpleNamespace(value="classification")))

    assert PredictionService().predict_output(
        manager=manager,
        target_encoder=None,
        predict_data=data,
        predict_mode="labels",
    ) == {"custom": data}


def test_prediction_service_applies_target_encoder_to_labels_and_target():
    class FakeEncoder:
        def inverse_transform(self, values):
            return np.asarray(values) + 10

    class FakeSolver:
        def predict(self, data):
            return np.array([0, 1])

    data = SimpleNamespace(
        target=np.array([1, 0]),
        task=SimpleNamespace(task_type=SimpleNamespace(value="classification")),
    )
    manager = SimpleNamespace(
        solver=FakeSolver(),
        condition_check=FakeConditionCheck(have_encoder=True),
    )

    result = PredictionService().predict_output(
        manager=manager,
        target_encoder=FakeEncoder(),
        predict_data=data,
        predict_mode="labels",
    )

    np.testing.assert_array_equal(result, np.array([10, 11]))
    np.testing.assert_array_equal(data.target, np.array([11, 10]))


def test_prediction_service_slices_forecasting_output_to_horizon():
    class FakeSolver:
        def predict(self, data):
            return np.arange(5)

    data = SimpleNamespace(
        task=SimpleNamespace(
            task_type=SimpleNamespace(value="ts_forecasting"),
            task_params=SimpleNamespace(forecast_length=2),
        )
    )
    manager = SimpleNamespace(solver=FakeSolver(), condition_check=FakeConditionCheck())

    result = PredictionService().predict_output(
        manager=manager,
        target_encoder=None,
        predict_data=data,
        predict_mode="labels",
    )

    np.testing.assert_array_equal(result, np.array([3, 4]))
