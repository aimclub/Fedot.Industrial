import json

import numpy as np
import pandas as pd
import pytest
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.kernel_learning.estimators.classifier import (
    KernelEnsembleClassifier,
)
from fedot_ind.core.kernel_learning.estimators.regressor import KernelEnsembleRegressor
from fedot_ind.core.models.pdl import (
    PairwiseDifferenceClassifier,
    PairwiseDifferenceRegressor,
)
from fedot_ind.integration.fedot import (
    DataProfile,
    IntegrationContractError,
    IntegrationErrorCode,
    IntegrationTask,
    ModelExecutionPlan,
    PredictionMode,
    create_supervised_runtime,
)


CLASSIFICATION_FEATURES = np.array(
    [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]]
)
CLASSIFICATION_TARGET = np.array([0, 0, 1, 1])
REGRESSION_FEATURES = np.arange(6, dtype=float).reshape(-1, 1)
REGRESSION_TARGET = 2.0 * REGRESSION_FEATURES.reshape(-1) + 1.0


def test_model_execution_plan_owns_an_immutable_parameter_snapshot():
    source = {"generator_names": ["identity"], "nested": {"alpha": 1.0}}
    plan = ModelExecutionPlan(
        profile=DataProfile.TENSOR,
        task=IntegrationTask.CLASSIFICATION,
        operation_name=" kernel_ensemble_clf ",
        parameters=source,
    )
    source["generator_names"].append("shapelet_extractor")
    source["nested"]["alpha"] = 2.0

    assert plan.operation_name == "kernel_ensemble_clf"
    assert plan.parameters["generator_names"] == ("identity",)
    assert plan.parameters["nested"]["alpha"] == 1.0
    with pytest.raises(TypeError):
        plan.parameters["nested"]["alpha"] = 3.0
    assert json.loads(json.dumps(plan.to_dict())) == plan.to_dict()
    assert plan.runtime_parameters()["generator_names"] == ["identity"]


def test_supervised_runtime_rejects_legacy_and_unsupported_prediction_modes():
    with pytest.raises(IntegrationContractError) as profile_error:
        create_supervised_runtime(
            profile="legacy",
            task="classification",
            operation_name="pdl_clf",
        )
    assert profile_error.value.code is IntegrationErrorCode.UNSUPPORTED_OPERATION

    with pytest.raises(IntegrationContractError) as operation_error:
        create_supervised_runtime(
            profile="tensor",
            task="classification",
            operation_name="missing_model",
        )
    assert operation_error.value.code is IntegrationErrorCode.UNSUPPORTED_OPERATION

    runtime = create_supervised_runtime(
        profile="tensor",
        task="regression",
        operation_name="pdl_reg",
        parameters={"model": "dtreg", "max_pairs": 100, "random_state": 42},
    ).fit(REGRESSION_FEATURES, REGRESSION_TARGET)
    with pytest.raises(IntegrationContractError) as mode_error:
        runtime.predict(REGRESSION_FEATURES, PredictionMode.PROBABILITIES)
    assert mode_error.value.code is IntegrationErrorCode.UNSUPPORTED_OUTPUT_MODE
    runtime.close()


def test_pdl_classifier_tensor_runtime_matches_direct_model_and_preserves_index():
    parameters = {"model": "dt", "max_pairs": 100, "random_state": 42}
    frame = pd.DataFrame(
        CLASSIFICATION_FEATURES,
        columns=["left", "right"],
        index=["a", "b", "c", "d"],
    )
    direct = PairwiseDifferenceClassifier(OperationParameters(**parameters)).fit(
        CLASSIFICATION_FEATURES,
        CLASSIFICATION_TARGET,
    )
    runtime = create_supervised_runtime(
        profile="tensor",
        task="classification",
        operation_name="pdl_clf",
        parameters=parameters,
    ).fit(frame, CLASSIFICATION_TARGET)

    probabilities = runtime.predict(frame, PredictionMode.FULL_PROBABILITIES)
    labels = runtime.predict(frame, PredictionMode.LABELS)

    np.testing.assert_allclose(
        probabilities.values,
        direct.predict_proba(CLASSIFICATION_FEATURES),
    )
    np.testing.assert_array_equal(
        labels.values.reshape(-1),
        direct.predict(CLASSIFICATION_FEATURES).reshape(-1),
    )
    assert probabilities.idx.tolist() == ["a", "b", "c", "d"]
    assert probabilities.classes.tolist() == [0, 1]
    runtime.close()


@pytest.mark.parametrize(
    "target",
    [
        np.array(["negative", "negative", "positive", "positive"]),
        np.array([10, 10, 30, 30]),
    ],
)
def test_tensor_runtime_decodes_original_class_labels(target):
    parameters = {"model": "dt", "max_pairs": 100, "random_state": 42}
    direct = PairwiseDifferenceClassifier(OperationParameters(**parameters)).fit(
        CLASSIFICATION_FEATURES,
        target,
    )
    runtime = create_supervised_runtime(
        profile="tensor",
        task="classification",
        operation_name="pdl_clf",
        parameters=parameters,
    ).fit(CLASSIFICATION_FEATURES, target)

    result = runtime.predict(CLASSIFICATION_FEATURES, PredictionMode.LABELS)

    np.testing.assert_array_equal(
        result.values.reshape(-1),
        direct.predict(CLASSIFICATION_FEATURES).reshape(-1),
    )
    assert result.classes.tolist() == np.unique(target).tolist()
    runtime.close()


def test_pdl_regressor_tensor_runtime_matches_direct_model():
    parameters = {"model": "dtreg", "max_pairs": 100, "random_state": 42}
    direct = PairwiseDifferenceRegressor(OperationParameters(**parameters)).fit(
        REGRESSION_FEATURES,
        REGRESSION_TARGET,
    )
    runtime = create_supervised_runtime(
        profile="tensor",
        task="regression",
        operation_name="pdl_reg",
        parameters=parameters,
    ).fit(REGRESSION_FEATURES, REGRESSION_TARGET)
    predict = np.array([[6.0], [7.0]])

    result = runtime.predict(predict)

    np.testing.assert_allclose(result.values, direct.predict(predict))
    runtime.close()


def test_kernel_classifier_tensor_runtime_matches_direct_model():
    parameters = {
        "generator_names": ["identity"],
        "kernel": "linear",
        "normalize": None,
        "center": False,
        "probability": True,
        "random_state": 42,
        "kernel_cache_enabled": False,
    }
    direct = KernelEnsembleClassifier(**parameters).fit(
        CLASSIFICATION_FEATURES,
        CLASSIFICATION_TARGET,
    )
    runtime = create_supervised_runtime(
        profile="tensor",
        task="classification",
        operation_name="kernel_ensemble_clf",
        parameters=parameters,
    ).fit(CLASSIFICATION_FEATURES, CLASSIFICATION_TARGET)

    result = runtime.predict(
        CLASSIFICATION_FEATURES,
        PredictionMode.FULL_PROBABILITIES,
    )

    np.testing.assert_allclose(
        result.values,
        direct.predict_proba(CLASSIFICATION_FEATURES),
    )
    runtime.close()


def test_kernel_regressor_tensor_runtime_matches_direct_model():
    parameters = {
        "generator_names": ["identity"],
        "kernel": "linear",
        "normalize": None,
        "center": False,
        "alpha": 1.0,
        "head_type": "kernel_ridge",
        "kernel_cache_enabled": False,
    }
    direct = KernelEnsembleRegressor(**parameters).fit(
        REGRESSION_FEATURES,
        REGRESSION_TARGET,
    )
    runtime = create_supervised_runtime(
        profile="tensor",
        task="regression",
        operation_name="kernel_ensemble_reg",
        parameters=parameters,
    ).fit(REGRESSION_FEATURES, REGRESSION_TARGET)
    predict = np.array([[6.0], [7.0]])

    result = runtime.predict(predict)

    np.testing.assert_allclose(result.values, direct.predict(predict))
    runtime.close()
