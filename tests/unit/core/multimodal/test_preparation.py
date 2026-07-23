import numpy as np
import pytest
import torch

from benchmark.industrial.core import ClassificationDatasetRecord
from fedot_ind.core.multimodal import (
    DEFAULT_STAT_FEATURES,
    MultimodalDatasetPreparer,
    MultimodalModality,
    NormalizationMethod,
    PreparationConfig,
    StatisticalFeature,
)
from fedot_ind.core.multimodal.configs import normalization_policy_from_steps


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


def test_normalization_policy_from_steps_returns_named_and_custom_policies():
    assert normalization_policy_from_steps(()) == "none"
    assert (
        normalization_policy_from_steps(
            (
                NormalizationMethod.imputation,
                NormalizationMethod.feature_standardization,
            )
        )
        == "train_mean_imputation_then_train_mean_std"
    )
    assert (
        normalization_policy_from_steps((NormalizationMethod.image_standardization,))
        == "train_image_standardization"
    )
    assert (
        normalization_policy_from_steps(
            (
                NormalizationMethod.log1p,
                NormalizationMethod.image_standardization,
            )
        )
        == "log1p_then_train_image_standardization"
    )
    assert (
        normalization_policy_from_steps((NormalizationMethod.log1p,))
        == "log1p"
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"transformation_config": {"unknown": {}}},
            "Unsupported modality key",
        ),
        (
            {
                "transformation_config": {
                    "raw": {
                        "per_sample_z_normalize": True,
                        "per_sample_z_normalize_eps": 0,
                    }
                }
            },
            "must be positive",
        ),
        (
            {
                "normalization_config": {
                    MultimodalModality.raw: (NormalizationMethod.log1p,)
                }
            },
            "Raw modality is not normalized",
        ),
        (
            {
                "transformation_config": {
                    "stats": {"feature_names": ("not_a_feature",)}
                }
            },
            "not_a_feature",
        ),
    ],
)
def test_preparation_config_rejects_invalid_contracts(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PreparationConfig(**kwargs)


def test_preparation_config_metadata_uses_resolved_transform_params():
    config = PreparationConfig(
        transformation_config={
            "raw": {"per_sample_z_normalize": True},
            "stats": {"feature_names": ("mean", "std")},
        }
    )
    resolved_stats_params = {"feature_names": ("mean",), "torch_device": torch.device("cpu")}

    metadata = config.metadata(
        torch.device("cpu"),
        transform_params={MultimodalModality.stats: resolved_stats_params},
    )

    assert metadata["normalization"][MultimodalModality.raw] == "per_sample_z_norm"
    assert metadata["transform_params"][MultimodalModality.stats] == resolved_stats_params
    assert metadata["preparation_config"]["transformation_config"]["stats"] == resolved_stats_params


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


def test_preparer_rejects_loader_without_load_data():
    preparer = MultimodalDatasetPreparer()

    with pytest.raises(TypeError, match="load_data"):
        preparer.prepare_from_loader(object())


def test_preparer_transform_requires_fit():
    preparer = MultimodalDatasetPreparer()

    with pytest.raises(ValueError, match="must be fitted"):
        preparer.transform(np.zeros((2, 4)), np.array([0, 1]))


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
        bundle.modalities[MultimodalModality.raw].cpu(),
        torch.tensor([[[1.0, 2.0, 3.0]], [[5.0, 5.0, 5.0]]]),
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.stats].cpu(),
        torch.tensor([[2.0, 0.8165], [5.0, 0.0]]),
        atol=1e-4,
    )
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "none"


def test_preparer_raises_on_unknown_labels_during_transform():
    config = PreparationConfig(
        transformation_config={
            "raw": {"per_sample_z_normalize": False},
        }
    )
    preparer = MultimodalDatasetPreparer(config=config)
    preparer.fit(np.zeros((2, 4)), np.array(["known", "known"]))

    with pytest.raises(ValueError, match="Unknown target labels"):
        preparer.transform(np.zeros((1, 4)), np.array(["new"]))


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
        bundle.modalities[MultimodalModality.raw][0].mean(dim=-1).cpu(),
        torch.zeros(1),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw][0].std(dim=-1, unbiased=False).cpu(),
        torch.ones(1),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.raw][1].cpu(),
        torch.zeros(1, 3),
        atol=1e-6,
    )
    assert torch.allclose(
        bundle.modalities[MultimodalModality.stats].cpu(),
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


def test_preparer_builds_mtf_modality_and_records_resolved_params():
    config = PreparationConfig(
        normalization_config={},
        transformation_config={
            "raw": {"per_sample_z_normalize": False},
            "mtf": {"image_size": 0.5, "n_bins": 3, "strategy": "uniform"},
        },
    )
    preparer = MultimodalDatasetPreparer(config=config)

    bundle = preparer.fit_transform(
        np.arange(2 * 8, dtype=float).reshape(2, 8),
        np.array([0, 1]),
    )

    assert bundle.modalities[MultimodalModality.mtf].shape == (2, 1, 4, 4)
    assert bundle.metadata["transform_params"][MultimodalModality.mtf]["n_bins"] == 3
    assert "mtf" in bundle.metadata["preparation_config"]["transformation_config"]


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
