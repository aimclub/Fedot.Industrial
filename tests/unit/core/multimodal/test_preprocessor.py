import pytest
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality, NormalizationMethod
from fedot_ind.core.multimodal.preprocessor import MultimodalPreprocessor


def make_bundle(
    *,
    raw: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    gaf: torch.Tensor | None = None,
    stft: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
) -> MultimodalDataBundle:
    modalities = {}
    if raw is not None:
        modalities[MultimodalModality.raw] = raw
    if stats is not None:
        modalities[MultimodalModality.stats] = stats
    if gaf is not None:
        modalities[MultimodalModality.gaf] = gaf
    if stft is not None:
        modalities[MultimodalModality.stft] = stft
    return MultimodalDataBundle(modalities=modalities, target=target)


def test_multimodal_preprocessor_keeps_raw_unchanged():
    raw = torch.tensor(
        [
            [[-1.0, 0.0, 1.0]],
            [[1.0, 0.0, -1.0]],
        ],
    )
    target = torch.tensor([0, 1])
    source = make_bundle(raw=raw, target=target)

    preprocessor = MultimodalPreprocessor()
    bundle = preprocessor.fit_transform(source)

    assert isinstance(bundle, MultimodalDataBundle)
    assert bundle.n_samples == 2
    assert bundle.target is target
    assert bundle.metadata["modalities"] == [MultimodalModality.raw]
    assert bundle.metadata["shapes"][MultimodalModality.raw] == (2, 1, 3)
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "none"
    assert "fitted_statistics" not in bundle.metadata
    assert preprocessor.fitted_statistics_ == {}
    assert torch.equal(bundle.modalities[MultimodalModality.raw], raw)


def test_multimodal_preprocessor_standardizes_gaf_images_with_train_statistics():
    train_gaf = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
    test_gaf = train_gaf + 100.0
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.gaf: [NormalizationMethod.image_standardization],
        }
    )

    train_bundle = preprocessor.fit_transform(make_bundle(gaf=train_gaf))
    test_bundle = preprocessor.transform(make_bundle(gaf=test_gaf))

    train_normalized = train_bundle.modalities[MultimodalModality.gaf]
    test_normalized = test_bundle.modalities[MultimodalModality.gaf]
    stats = preprocessor.fitted_statistics_["gaf"]["image_standardization"]

    assert stats["mean"].shape == (1, 1, 1, 1)
    assert stats["std"].shape == (1, 1, 1, 1)
    assert torch.allclose(train_normalized.mean(dim=(0, 2, 3)), torch.zeros(1), atol=1e-6)
    assert torch.allclose(
        train_normalized.std(dim=(0, 2, 3), unbiased=False),
        torch.ones(1),
        atol=1e-6,
    )
    expected_test = torch.nan_to_num(((test_gaf - stats["mean"]) / stats["std"]).float())
    assert torch.allclose(test_normalized, expected_test)
    assert "fitted_statistics" not in train_bundle.metadata


def test_multimodal_preprocessor_applies_stft_log1p_before_image_standardization():
    train_stft = torch.arange(1, 1 + 2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
    test_stft = train_stft + 10.0
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.stft: [
                NormalizationMethod.log1p,
                NormalizationMethod.image_standardization,
            ],
        }
    )

    train_bundle = preprocessor.fit_transform(make_bundle(stft=train_stft))
    test_bundle = preprocessor.transform(make_bundle(stft=test_stft))

    stats = preprocessor.fitted_statistics_["stft"]["image_standardization"]
    log_train = torch.log1p(train_stft.float().clamp_min(0))
    expected_train = torch.nan_to_num(((log_train - stats["mean"]) / stats["std"]).float())
    log_test = torch.log1p(test_stft.float().clamp_min(0))
    expected_test = torch.nan_to_num(((log_test - stats["mean"]) / stats["std"]).float())

    assert preprocessor.fitted_statistics_["stft"]["steps"] == [
        "log1p",
        "image_standardization",
    ]
    assert torch.allclose(train_bundle.modalities[MultimodalModality.stft], expected_train)
    assert torch.allclose(test_bundle.modalities[MultimodalModality.stft], expected_test)


def test_multimodal_preprocessor_does_not_update_statistics_on_transform():
    train_gaf = torch.arange(2 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 1, 2, 2)
    test_gaf = train_gaf + 100.0
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.gaf: [NormalizationMethod.image_standardization],
        }
    ).fit(make_bundle(gaf=train_gaf))
    train_mean = preprocessor.fitted_statistics_["gaf"]["image_standardization"]["mean"].clone()
    train_std = preprocessor.fitted_statistics_["gaf"]["image_standardization"]["std"].clone()

    preprocessor.transform(make_bundle(gaf=test_gaf))

    stats = preprocessor.fitted_statistics_["gaf"]["image_standardization"]
    assert torch.equal(stats["mean"], train_mean)
    assert torch.equal(stats["std"], train_std)


def test_multimodal_preprocessor_supports_configured_steps():
    stft = torch.arange(1, 1 + 2 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 1, 2, 2)
    preprocessor = MultimodalPreprocessor(
        normalization_config={MultimodalModality.stft: [NormalizationMethod.log1p]},
    )

    bundle = preprocessor.fit_transform(make_bundle(stft=stft))

    assert torch.allclose(
        bundle.modalities[MultimodalModality.stft],
        torch.log1p(stft.float().clamp_min(0)),
    )
    assert "image_standardization" not in preprocessor.fitted_statistics_["stft"]


def test_multimodal_preprocessor_imputes_stats_with_train_column_means():
    train_stats = torch.tensor(
        [
            [1.0, float("nan")],
            [3.0, 4.0],
        ]
    )
    test_stats = torch.tensor(
        [
            [float("inf"), 10.0],
            [5.0, float("nan")],
        ]
    )
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.stats: [NormalizationMethod.imputation],
        }
    )

    train_bundle = preprocessor.fit_transform(make_bundle(stats=train_stats))
    test_bundle = preprocessor.transform(make_bundle(stats=test_stats))
    column_mean = preprocessor.fitted_statistics_["stats"]["imputation"]["column_mean"]

    assert torch.allclose(column_mean, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(
        train_bundle.modalities[MultimodalModality.stats],
        torch.tensor([[1.0, 4.0], [3.0, 4.0]]),
    )
    assert torch.allclose(
        test_bundle.modalities[MultimodalModality.stats],
        torch.tensor([[2.0, 10.0], [5.0, 4.0]]),
    )


def test_multimodal_preprocessor_standardizes_stats_per_feature_with_train_statistics():
    train_stats = torch.tensor([[0.0, 2.0], [2.0, 6.0]])
    test_stats = torch.tensor([[4.0, 10.0]])
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.stats: [NormalizationMethod.feature_standardization],
        }
    )

    train_bundle = preprocessor.fit_transform(make_bundle(stats=train_stats))
    test_bundle = preprocessor.transform(make_bundle(stats=test_stats))
    stats = preprocessor.fitted_statistics_["stats"]["feature_standardization"]

    assert torch.allclose(stats["mean"], torch.tensor([[1.0, 4.0]]))
    assert torch.allclose(stats["std"], torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(
        train_bundle.modalities[MultimodalModality.stats],
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    expected_test = torch.tensor([[3.0, 3.0]])
    assert torch.allclose(test_bundle.modalities[MultimodalModality.stats], expected_test)


def test_multimodal_preprocessor_applies_stats_imputation_before_feature_standardization():
    train_stats = torch.tensor(
        [
            [1.0, float("nan")],
            [3.0, 5.0],
            [5.0, 1.0],
        ]
    )
    test_stats = torch.tensor([[float("inf"), 9.0]])
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.stats: [
                NormalizationMethod.imputation,
                NormalizationMethod.feature_standardization,
            ],
        }
    )

    train_bundle = preprocessor.fit_transform(make_bundle(stats=train_stats))
    test_bundle = preprocessor.transform(make_bundle(stats=test_stats))

    imputed_train = torch.tensor([[1.0, 3.0], [3.0, 5.0], [5.0, 1.0]])
    mean = imputed_train.mean(dim=0, keepdim=True)
    std = imputed_train.std(dim=0, unbiased=False, keepdim=True)
    expected_train = (imputed_train - mean) / std
    imputed_test = torch.tensor([[3.0, 9.0]])
    expected_test = (imputed_test - mean) / std

    assert torch.allclose(
        train_bundle.modalities[MultimodalModality.stats],
        expected_train,
    )
    assert torch.allclose(
        test_bundle.modalities[MultimodalModality.stats],
        expected_test,
    )


def test_multimodal_preprocessor_does_not_update_stats_statistics_on_transform():
    train_stats = torch.tensor([[0.0, 2.0], [2.0, 6.0]])
    test_stats = torch.tensor([[4.0, 10.0]])
    preprocessor = MultimodalPreprocessor(
        normalization_config={
            MultimodalModality.stats: [NormalizationMethod.feature_standardization],
        }
    ).fit(make_bundle(stats=train_stats))
    train_mean = preprocessor.fitted_statistics_["stats"]["feature_standardization"]["mean"].clone()
    train_std = preprocessor.fitted_statistics_["stats"]["feature_standardization"]["std"].clone()

    preprocessor.transform(make_bundle(stats=test_stats))

    stats = preprocessor.fitted_statistics_["stats"]["feature_standardization"]
    assert torch.equal(stats["mean"], train_mean)
    assert torch.equal(stats["std"], train_std)


def test_multimodal_preprocessor_rejects_unfitted_transform():
    with pytest.raises(ValueError, match="must be fitted"):
        MultimodalPreprocessor().transform(make_bundle(raw=torch.randn(2, 1, 4)))


def test_multimodal_preprocessor_rejects_non_bundle_input():
    with pytest.raises(TypeError, match="MultimodalDataBundle"):
        MultimodalPreprocessor().fit(torch.randn(2, 1, 4))


def test_multimodal_preprocessor_requires_configured_modalities():
    with pytest.raises(ValueError, match="required modalities"):
        MultimodalPreprocessor(
            normalization_config={
                MultimodalModality.gaf: [NormalizationMethod.image_standardization],
            }
        ).fit(make_bundle(raw=torch.randn(2, 1, 4)))
