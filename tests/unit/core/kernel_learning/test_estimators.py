import numpy as np

from fedot_ind.core.kernel_learning import KernelEnsembleClassifier, KernelEnsembleRegressor


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
