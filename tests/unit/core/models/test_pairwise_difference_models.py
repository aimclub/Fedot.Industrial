from fedot_ind.core.models.pdl import PairwiseDifferenceClassifier, PairwiseDifferenceRegressor
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
import numpy as np
import pytest

pytest.importorskip("fedot")


def _classification_input(features, target):
    return InputData(
        idx=np.arange(len(features)),
        features=np.asarray(features, dtype=float),
        target=np.asarray(target, dtype=object),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def _regression_input(features, target):
    return InputData(
        idx=np.arange(len(features)),
        features=np.asarray(features, dtype=float),
        target=np.asarray(target, dtype=float),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
    )


def test_pdl_classifier_predicts_probabilities_and_decodes_original_labels():
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
            [4.0, 4.0],
            [4.1, 4.0],
        ]
    )
    target = np.array(["alpha", "alpha", "beta", "beta", "gamma", "gamma"])
    model = PairwiseDifferenceClassifier(
        OperationParameters(
            model="rf",
            backend="numpy",
            max_pairs=100,
            n_estimators=20,
            random_state=42,
        )
    )

    model.fit(_classification_input(features, target))
    proba = model.predict_proba(features)
    labels = model.predict(features)
    fit_prediction = model.predict_for_fit(_classification_input(features, target))

    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))
    assert labels.shape == (6, 1)
    assert set(labels.reshape(-1)).issubset({"alpha", "beta", "gamma"})
    assert fit_prediction.shape == (6, 3)
    assert model.get_diagnostics()["n_classes"] == 3


def test_pdl_regressor_recovers_simple_linear_target_with_ridge_base_model():
    features = np.arange(6, dtype=float).reshape(-1, 1)
    target = 2.0 * features.reshape(-1) + 1.0
    model = PairwiseDifferenceRegressor(
        {
            "model": "ridge",
            "backend": "numpy",
            "max_pairs": 100,
            "alpha": 1e-8,
        }
    )

    model.fit(_regression_input(features, target))
    prediction = model.predict(np.array([[6.0], [7.0]]))

    assert prediction.shape == (2,)
    np.testing.assert_allclose(prediction, np.array([13.0, 15.0]), atol=1e-4)
    assert model.predict_for_fit(features).shape == (6,)
    assert model.get_diagnostics()["task"] == "regression"
