from copy import deepcopy

import pytest
import torch

from fedot_ind.core.multimodal import (
    MultimodalDataBundle,
    MultimodalModality,
    MultimodalPreprocessor,
)


def test_multimodal_preprocessor_builds_raw_bundle_metadata():
    X = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 3.5, 4.0, 4.5],
        ]
    )
    y = torch.tensor([0, 1, 0])

    bundle = MultimodalPreprocessor().fit_transform(X, y)

    assert isinstance(bundle, MultimodalDataBundle)
    assert bundle.n_samples == 3
    assert bundle.target.tolist() == [0, 1, 0]
    assert bundle.metadata["modalities"] == [MultimodalModality.raw]
    assert bundle.metadata["shapes"][MultimodalModality.raw] == (3, 1, 4)
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "per_sample_z_norm"
    assert bundle.metadata["transform_params"][MultimodalModality.raw]["eps"] == 1e-8
    assert bundle.metadata["fitted_statistics"]["raw"]["train_input_shape"] == (3, 1, 4)

    raw = bundle.modalities[MultimodalModality.raw]
    assert torch.allclose(raw.mean(dim=-1), torch.zeros(3, 1), atol=1e-6)
    assert torch.allclose(raw.std(dim=-1, unbiased=False), torch.ones(3, 1), atol=1e-6)


def test_multimodal_preprocessor_does_not_update_statistics_on_transform():
    train = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    test = torch.tensor([[100.0, 101.0, 102.0]])
    preprocessor = MultimodalPreprocessor().fit(train)
    train_statistics = deepcopy(preprocessor.fitted_statistics_)

    bundle = preprocessor.transform(test)

    assert preprocessor.fitted_statistics_ == train_statistics
    assert bundle.metadata["fitted_statistics"] == train_statistics
    raw = bundle.modalities[MultimodalModality.raw]
    assert raw.shape == (1, 1, 3)
    assert torch.allclose(raw.mean(dim=-1), torch.zeros(1, 1), atol=1e-6)


def test_multimodal_preprocessor_handles_constant_and_short_series():
    X = torch.tensor([[5.0], [5.0]])

    bundle = MultimodalPreprocessor().fit_transform(X)
    raw = bundle.modalities[MultimodalModality.raw]

    assert raw.shape == (2, 1, 1)
    assert torch.all(torch.isfinite(raw))
    assert torch.allclose(raw, torch.zeros_like(raw))


def test_multimodal_preprocessor_preserves_multichannel_layout():
    X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    bundle = MultimodalPreprocessor().fit_transform(X)
    raw = bundle.modalities[MultimodalModality.raw]

    assert raw.shape == (2, 3, 4)
    assert bundle.metadata["shapes"][MultimodalModality.raw] == (2, 3, 4)
    assert torch.allclose(raw.mean(dim=-1), torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(raw.std(dim=-1, unbiased=False), torch.ones(2, 3), atol=1e-6)


def test_multimodal_preprocessor_checks_target_sample_consistency():
    with pytest.raises(ValueError, match="Target and modalities"):
        MultimodalPreprocessor().fit_transform(
            torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
            torch.tensor([0]),
        )


def test_multimodal_preprocessor_rejects_unfitted_transform():
    with pytest.raises(ValueError, match="must be fitted"):
        MultimodalPreprocessor().transform(torch.tensor([[0.0, 1.0]]))


def test_multimodal_preprocessor_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="X must be torch.Tensor"):
        MultimodalPreprocessor().fit([[0.0, 1.0]])

    with pytest.raises(TypeError, match="Target must be torch.Tensor"):
        MultimodalPreprocessor().fit_transform(
            torch.tensor([[0.0, 1.0]]),
            [0],
        )
