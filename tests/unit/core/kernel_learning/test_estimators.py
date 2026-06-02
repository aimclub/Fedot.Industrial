import numpy as np

from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleForecaster, KernelEnsembleRegressor
from fedot_ind.core.kernel_learning.contracts import KernelBundle, KernelSelectionReport


def test_kernel_ensemble_classifier_predicts_and_returns_probabilities():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    y = np.array(["a", "a", "b", "b"])

    model = KernelEnsembleClassifier(generator_names=("identity",), kernel="linear", C=10.0)
    model.fit(X, y)
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)

    assert prediction.shape == (4,)
    assert probabilities.shape == (4, 2)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert model.selected_generators_ == ("identity",)
    assert model.important_generators_ == ("identity",)
    assert model.kernel_importance_.selected_generators == ("identity",)


def test_kernel_ensemble_regressor_predicts_stable_shape():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    model = KernelEnsembleRegressor(
        generator_names=("identity",),
        kernel="linear",
        alpha=1e-6,
    )
    model.fit(X, y)
    prediction = model.predict(np.array([[5.0], [6.0]]))

    assert prediction.shape == (2,)
    assert np.all(np.isfinite(prediction))
    assert prediction[1] > prediction[0]


def test_kernel_ensemble_forecaster_predicts_multi_horizon_shape():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    y = np.array(
        [
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ]
    )

    model = KernelEnsembleForecaster(
        generator_names=("identity",),
        kernel="linear",
        alpha=1e-6,
        forecast_horizon=2,
    )
    model.fit(X, y)
    prediction = model.predict(np.array([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]))

    assert prediction.shape == (2, 2)
    assert np.all(np.isfinite(prediction))
    assert model.selection_report_.task_type == "forecasting"


def test_kernel_mixing_still_uses_selector_weights_not_importance_threshold():
    model = KernelEnsembleClassifier(importance_threshold=0.8)
    model.selection_report_ = KernelSelectionReport(
        generator_names=("a", "b"),
        weights=(0.6, 0.4),
        selected_generators=("a", "b"),
        selected_weights=(0.6, 0.4),
        scores={"a": 0.6, "b": 0.4},
        alignments={"a": 0.6, "b": 0.4},
        complexities={"a": 0.0, "b": 0.0},
        redundancies={"a": 0.0, "b": 0.0},
        task_type="classification",
    )
    model.kernel_bundles_ = [
        KernelBundle(name="a", train_kernel=np.ones((2, 2))),
        KernelBundle(name="b", train_kernel=np.eye(2) * 10.0),
    ]

    combined = model._combine_train_kernels()

    np.testing.assert_allclose(combined, np.array([[4.6, 0.6], [0.6, 4.6]]))
