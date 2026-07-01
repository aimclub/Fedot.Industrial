import numpy as np
import pytest
import torch

from benchmark.industrial.core import ClassificationDatasetRecord
from fedot_ind.core.multimodal import (
    DEFAULT_STAT_FEATURES,
    MultimodalDatasetPreparer,
    MultimodalModality,
    PreparationConfig,
    StatisticalFeature,
)


def make_record() -> ClassificationDatasetRecord:
    train_features = tuple(
        tuple(float(value) for value in row)
        for row in np.arange(4 * 8, dtype=float).reshape(4, 8)
    )
    test_features = tuple(
        tuple(float(value) for value in row)
        for row in (np.arange(2 * 8, dtype=float).reshape(2, 8) + 100.0)
    )
    return ClassificationDatasetRecord(
        benchmark="synthetic",
        dataset_name="TinySynthetic",
        subset="default",
        train_features=train_features,
        train_target=("a", "b", "a", "b"),
        test_features=test_features,
        test_target=("a", "b"),
    )


def test_preparer_builds_default_modalities_from_classification_record():
    record = make_record()
    preparer = MultimodalDatasetPreparer()

    train_bundle, test_bundle = preparer.prepare_classification_record(record)
    repeated_test_bundle = preparer.transform(record.test_features, record.test_target)

    assert train_bundle.n_samples == 4
    assert test_bundle.n_samples == 2
    assert set(train_bundle.modalities) == {
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    }
    assert train_bundle.target.tolist() == [0, 1, 0, 1]
    assert test_bundle.target.tolist() == [0, 1]
    assert train_bundle.metadata["source"]["kind"] == "ClassificationDatasetRecord"
    assert train_bundle.metadata["normalization"][MultimodalModality.raw] == "none"
    assert train_bundle.metadata["transform_params"][MultimodalModality.stats][
        "feature_names"
    ] == DEFAULT_STAT_FEATURES
    assert train_bundle.metadata["transform_params"][MultimodalModality.stft][
        "window_size"
    ] == 8

    for modality in test_bundle.modalities:
        if modality is MultimodalModality.stats:
            assert torch.allclose(
                test_bundle.modalities[modality],
                repeated_test_bundle.modalities[modality],
                atol=2e-1,
            )
        else:
            assert torch.allclose(
                test_bundle.modalities[modality],
                repeated_test_bundle.modalities[modality],
            )


def test_preparer_does_not_update_statistics_on_record_test_transform():
    record = make_record()
    preparer = MultimodalDatasetPreparer()
    preparer.prepare_classification_record(record)
    stats_before = {
        modality: {
            step: {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in step_stats.items()
            }
            for step, step_stats in modality_stats.items()
            if isinstance(step_stats, dict)
        }
        for modality, modality_stats in preparer.preprocessor_.fitted_statistics_.items()
    }

    preparer.transform(record.test_features, record.test_target)

    for modality, modality_stats in stats_before.items():
        for step, step_stats in modality_stats.items():
            current = preparer.preprocessor_.fitted_statistics_[modality][step]
            for key, expected in step_stats.items():
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(current[key], expected)
                else:
                    assert current[key] == expected


def test_preparer_builds_modalities_from_dataloader_like_object():
    class FakeLoader:
        dataset_name = "FakeTiny"

        def load_data(self):
            train_x = np.arange(3 * 6, dtype=float).reshape(3, 6)
            test_x = np.arange(2 * 6, dtype=float).reshape(2, 6)
            return (train_x, np.array([0, 1, 0])), (test_x, np.array([1, 0]))

    config = PreparationConfig(
        transformation_config={
            "raw": {"per_sample_z_normalize": False},
            "stats": {"feature_names": ("mean", "std")},
        }
    )
    preparer = MultimodalDatasetPreparer(config=config)

    train_bundle, test_bundle = preparer.prepare_from_loader(FakeLoader())

    assert train_bundle.metadata["source"]["kind"] == "DataLoader"
    assert train_bundle.metadata["source"]["dataset_name"] == "FakeTiny"
    assert train_bundle.modalities[MultimodalModality.raw].shape == (3, 1, 6)
    assert test_bundle.modalities[MultimodalModality.stats].shape[0] == 2


def test_preparer_keeps_raw_unchanged_and_builds_stats_from_input():
    config = PreparationConfig(
        normalization_config={},
        transformation_config={
            "raw": {"per_sample_z_normalize": False},
            "stats": {"feature_names": ("mean", "std")},
        },
    )
    preparer = MultimodalDatasetPreparer(config=config)
    train_x = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ]
    )

    bundle = preparer.fit_transform(train_x, np.array([0, 1]))

    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw],
        torch.tensor([[[1.0, 2.0, 3.0]], [[5.0, 5.0, 5.0]]]),
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.stats],
        torch.tensor([[2.0, 0.8165], [5.0, 0.0]]),
        atol=1e-4,
    )
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "none"


def test_preparer_can_per_sample_z_normalize_before_building_modalities():
    config = PreparationConfig(
        normalization_config={},
        transformation_config={
            "raw": {"per_sample_z_normalize": True, "per_sample_z_normalize_eps": 1e-6},
            "stats": {"feature_names": ("mean", "std")},
        },
    )
    preparer = MultimodalDatasetPreparer(config=config)
    train_x = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 5.0, 5.0],
        ]
    )

    bundle = preparer.fit_transform(train_x, np.array([0, 1]))

    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw][0].mean(dim=-1),
        torch.zeros(1),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw][0].std(dim=-1, unbiased=False),
        torch.ones(1),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw][1],
        torch.zeros(1, 3),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.stats],
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        atol=1e-6,
    )
    assert (
        bundle.metadata["transform_params"][MultimodalModality.raw][
            "per_sample_z_normalize"
        ]
        is True
    )
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "per_sample_z_norm"


def test_preparer_handles_short_constant_multichannel_series():
    train_x = np.ones((3, 2, 2), dtype=float)
    test_x = np.full((2, 2, 2), 5.0, dtype=float)
    preparer = MultimodalDatasetPreparer()

    train_bundle, test_bundle = preparer.prepare_train_test(
        (train_x, np.array([0, 1, 0])),
        (test_x, np.array([1, 0])),
    )

    assert train_bundle.modalities[MultimodalModality.raw].shape == (3, 2, 2)
    assert train_bundle.modalities[MultimodalModality.gaf].shape == (3, 2, 2, 2)
    assert train_bundle.modalities[MultimodalModality.stats].shape == (
        3,
        2 * len(DEFAULT_STAT_FEATURES),
    )
    assert train_bundle.metadata["transform_params"][MultimodalModality.stft][
        "window_size"
    ] == 2
    assert torch.allclose(
        train_bundle.modalities[MultimodalModality.raw],
        torch.ones_like(train_bundle.modalities[MultimodalModality.raw]),
    )
    for bundle in (train_bundle, test_bundle):
        for tensor in bundle.modalities.values():
            assert torch.isfinite(tensor).all()


def test_preparer_rejects_inconsistent_target_size():
    preparer = MultimodalDatasetPreparer(
        config=PreparationConfig(
            transformation_config={"raw": {"per_sample_z_normalize": False}}
        )
    )

    with pytest.raises(ValueError, match="Target and modalities"):
        preparer.fit_transform(np.zeros((3, 4)), np.array([0, 1]))


def test_preparer_accepts_stats_features_from_enum():
    config = PreparationConfig(
        normalization_config={},
        transformation_config={
            "raw": {"per_sample_z_normalize": False},
            "stats": {
                "feature_names": (
                    StatisticalFeature.mean,
                    StatisticalFeature.std,
                )
            },
        },
    )
    preparer = MultimodalDatasetPreparer(config=config)
    bundle = preparer.fit_transform(np.array([[1.0, 2.0, 3.0]]), np.array([0]))
    assert bundle.modalities[MultimodalModality.stats].shape == (1, 2)
