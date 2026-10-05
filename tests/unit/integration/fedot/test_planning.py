"""Data planning and normalization contracts without the FEDOT runtime."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from fedot_ind.integration.fedot import (
    AxisLayout,
    IntegrationContractError,
    IntegrationErrorCode,
    build_data_plan,
    build_multimodal_plan,
    normalize_input_data,
    normalize_multimodal_input,
    validate_prediction_plan,
)


@pytest.mark.parametrize("positions", list(permutations(range(3))))
def test_axis_permutations_preserve_every_sample_channel_and_feature(positions):
    canonical = np.arange(30).reshape(2, 3, 5)
    source = np.moveaxis(canonical, (0, 1, 2), positions)
    plan = build_data_plan(
        source, profile="tensor", task="regression", stage="train",
        axes=AxisLayout(sample=positions[0], channel=positions[1], feature=positions[2]),
    )

    prepared = normalize_input_data(source, plan)

    np.testing.assert_array_equal(prepared.values, canonical)
    np.testing.assert_array_equal(prepared.idx, [0, 1])
    assert prepared.schema.dimensions == (3, 5)
    assert not np.shares_memory(prepared.values, source)
    assert not prepared.values.flags.writeable


@pytest.mark.parametrize("source", [None, [1, 2], np.array(1), np.empty((0, 2)), np.empty((2, 0))])
def test_empty_or_unsupported_data_is_rejected_before_planning(source):
    with pytest.raises(IntegrationContractError) as error:
        build_data_plan(source, profile="tensor", task="regression", stage="train")

    assert error.value.code is IntegrationErrorCode.INVALID_DATA


@pytest.mark.parametrize("columns", [["value", "value"], [1, "1"]])
def test_column_names_must_remain_unique_after_string_conversion(columns):
    source = pd.DataFrame([[1, 2]], columns=columns)

    with pytest.raises(IntegrationContractError) as error:
        build_data_plan(source, profile="tensor", task="regression", stage="train")

    assert error.value.code is IntegrationErrorCode.INVALID_DATA


@pytest.mark.parametrize("stage_pair", [("train", "train"), ("predict", "predict"), ("predict", "train")])
def test_prediction_validation_requires_training_then_prediction(stage_pair):
    plans = [build_data_plan(np.ones((2, 3)), profile="tensor", task="regression", stage=stage)
             for stage in stage_pair]

    with pytest.raises(IntegrationContractError) as error:
        validate_prediction_plan(*plans)

    assert error.value.code is IntegrationErrorCode.PLAN_MISMATCH
    assert error.value.context == dict(zip(("train_stage", "predict_stage"), stage_pair))


def test_prediction_cannot_change_task_even_when_features_match():
    train = build_data_plan(np.ones((2, 3)), profile="tensor", task="regression", stage="train")
    predict = build_data_plan(np.ones((1, 3)), profile="tensor", task="classification", stage="predict")

    with pytest.raises(IntegrationContractError) as error:
        validate_prediction_plan(train, predict)

    assert error.value.code is IntegrationErrorCode.PLAN_MISMATCH
    assert error.value.context["mismatches"] == {
        "task": {"train": "regression", "predict": "classification"},
    }


def test_normalization_rejects_features_changed_since_planning():
    source = pd.DataFrame({"left": [1, 2], "right": [3, 4]})
    plan = build_data_plan(source, profile="tensor", task="regression", stage="train")
    source.rename(columns={"right": "replacement"}, inplace=True)

    with pytest.raises(IntegrationContractError) as error:
        normalize_input_data(source, plan)

    assert error.value.code is IntegrationErrorCode.SCHEMA_MISMATCH
    assert plan.feature_schema.columns == ("left", "right")


@pytest.mark.parametrize("name", ["", " raw", "raw ", 1])
def test_multimodal_names_are_not_silently_trimmed_or_coerced(name):
    with pytest.raises(IntegrationContractError) as error:
        build_multimodal_plan({name: np.ones((2, 3))}, task="regression", stage="train")

    assert error.value.code is IntegrationErrorCode.INVALID_DATA


def test_multimodal_sample_count_mismatch_identifies_each_modality():
    with pytest.raises(IntegrationContractError) as error:
        build_multimodal_plan(
            {"raw": np.ones((2, 1, 4)), "stats": np.ones((3, 2))},
            task="regression", stage="train",
        )

    assert error.value.code is IntegrationErrorCode.LENGTH_MISMATCH
    assert error.value.context["sample_counts"] == {"raw": 2, "stats": 3}


def test_multimodal_normalization_rejects_a_modality_added_after_planning():
    source = {"raw": np.ones((2, 1, 4))}
    plan = build_multimodal_plan(source, task="regression", stage="train")
    source["stats"] = np.ones((2, 2))

    with pytest.raises(IntegrationContractError) as error:
        normalize_multimodal_input(source, plan)

    assert error.value.code is IntegrationErrorCode.MODALITY_MISMATCH
    assert error.value.context == {"planned": ["raw"], "observed": ["raw", "stats"]}
