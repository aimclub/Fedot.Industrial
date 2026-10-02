import json

import numpy as np
import pandas as pd
import pytest
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.integration.fedot import (
    DataStage,
    IntegrationContractError,
    IntegrationErrorCode,
    IntegrationTask,
    build_multimodal_plan,
    create_multimodal_tensor_data,
    normalize_multimodal_input,
    validate_multimodal_prediction_plan,
)


def test_multimodal_plan_is_deterministic_and_json_serializable():
    data = {
        "stats": pd.DataFrame({"mean": [1.0, 2.0]}, index=["a", "b"]),
        "raw": np.ones((2, 1, 4)),
    }

    first = build_multimodal_plan(
        data,
        task=IntegrationTask.CLASSIFICATION,
        stage=DataStage.TRAIN,
    )
    second = build_multimodal_plan(
        dict(reversed(tuple(data.items()))),
        task="classification",
        stage="train",
    )

    assert first == second
    assert tuple(first.modalities) == ("raw", "stats")
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()


def test_multimodal_normalization_requires_exact_sample_coordinates():
    plan = build_multimodal_plan(
        {
            "left": pd.DataFrame({"value": [1.0, 2.0]}, index=["a", "b"]),
            "right": pd.DataFrame({"value": [3.0, 4.0]}, index=["b", "a"]),
        },
        task="regression",
        stage="train",
    )

    with pytest.raises(IntegrationContractError) as error:
        normalize_multimodal_input(
            {
                "left": pd.DataFrame({"value": [1.0, 2.0]}, index=["a", "b"]),
                "right": pd.DataFrame({"value": [3.0, 4.0]}, index=["b", "a"]),
            },
            plan,
        )

    assert error.value.code is IntegrationErrorCode.INDEX_MISMATCH


@pytest.mark.parametrize(
    ("samples", "raw_width", "stats_width"),
    [(1, 3, 1), (3, 8, 2), (7, 5, 4)],
)
def test_multimodal_normalization_conserves_samples_shapes_and_indexes(
        samples,
        raw_width,
        stats_width,
):
    raw = np.arange(samples * raw_width, dtype=float).reshape(samples, 1, raw_width)
    stats = np.arange(samples * stats_width, dtype=float).reshape(samples, stats_width)
    plan = build_multimodal_plan(
        {"stats": stats, "raw": raw},
        task="classification",
        stage="train",
    )

    prepared = normalize_multimodal_input({"stats": stats, "raw": raw}, plan)

    assert prepared.sample_count == samples
    assert prepared.idx.tolist() == list(range(samples))
    assert prepared.modalities["raw"].values.shape == (samples, 1, raw_width)
    assert prepared.modalities["stats"].values.shape == (samples, stats_width)
    assert not prepared.modalities["raw"].values.flags.writeable
    assert not prepared.modalities["stats"].values.flags.writeable


@pytest.mark.parametrize(
    "prediction",
    [
        {"raw": np.ones((1, 1, 4))},
        {"raw": np.ones((1, 1, 5)), "stats": np.ones((1, 2))},
    ],
)
def test_multimodal_prediction_requires_same_modalities_and_schemas(prediction):
    train = build_multimodal_plan(
        {"raw": np.ones((2, 1, 4)), "stats": np.ones((2, 2))},
        task="classification",
        stage="train",
    )
    predict = build_multimodal_plan(
        prediction,
        task="classification",
        stage="predict",
    )

    with pytest.raises(IntegrationContractError) as error:
        validate_multimodal_prediction_plan(train, predict)

    assert error.value.code in {
        IntegrationErrorCode.MODALITY_MISMATCH,
        IntegrationErrorCode.SCHEMA_MISMATCH,
    }


def test_multimodal_bundle_bridge_builds_aligned_tensor_data_and_reuses_plans():
    train_bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.arange(24, dtype=torch.float32).reshape(3, 1, 8),
            MultimodalModality.stats: torch.arange(6, dtype=torch.float32).reshape(3, 2),
        },
        target=torch.tensor([0, 1, 0]),
    )
    train = create_multimodal_tensor_data(
        train_bundle,
        task="classification",
        stage="train",
        idx=np.array(["sample-a", "sample-b", "sample-c"]),
    )
    predict_bundle = train_bundle.without_target()
    predict = create_multimodal_tensor_data(
        predict_bundle,
        task="classification",
        stage="predict",
        train_plan=train.plan,
        from_data=train.modalities,
        idx=np.array(["future-a", "future-b", "future-c"]),
    )

    assert tuple(train.modalities) == ("raw", "stats")
    for tensor_data in train.modalities.values():
        assert tensor_data.idx.tolist() == ["sample-a", "sample-b", "sample-c"]
        assert tensor_data.target.shape[0] == 3
    for tensor_data in predict.modalities.values():
        assert tensor_data.idx.tolist() == ["future-a", "future-b", "future-c"]
        assert tensor_data.target is None


def test_multimodal_bundle_bridge_rejects_incomplete_fitted_sources():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.ones((2, 1, 4)),
            MultimodalModality.stats: torch.ones((2, 2)),
        },
        target=torch.tensor([0, 1]),
    )

    with pytest.raises(IntegrationContractError) as error:
        create_multimodal_tensor_data(
            bundle,
            task="classification",
            stage="train",
            from_data={"raw": object()},
        )

    assert error.value.code is IntegrationErrorCode.MODALITY_MISMATCH
