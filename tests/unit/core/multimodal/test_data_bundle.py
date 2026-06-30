import pytest
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality


def test_multimodal_data_bundle_builds_metadata():
    bundle = MultimodalDataBundle(
        modalities={
            MultimodalModality.raw: torch.randn(8, 1, 128),
            MultimodalModality.stats: torch.randn(8, 10),
            MultimodalModality.gaf: torch.randn(8, 1, 32, 32),
            MultimodalModality.stft: torch.randn(8, 1, 16, 20),
        },
        target=torch.randint(0, 2, size=(8,)),
    )

    assert bundle.n_samples == 8
    assert bundle.metadata["modalities"] == [
        MultimodalModality.raw,
        MultimodalModality.stats,
        MultimodalModality.gaf,
        MultimodalModality.stft,
    ]
    assert bundle.metadata["shapes"][MultimodalModality.raw] == (8, 1, 128)
    assert bundle.metadata["normalization"][MultimodalModality.raw] == "per_sample_z_norm"
    assert bundle.metadata["normalization"][MultimodalModality.stats] == "train_mean_std"
    assert bundle.metadata["device"] == torch.device("cpu")
    assert bundle.metadata["dtype"] == torch.float32


def test_multimodal_data_bundle_checks_sample_consistency():
    with pytest.raises(ValueError, match="same number of samples"):
        MultimodalDataBundle(
            modalities={
                MultimodalModality.raw: torch.randn(8, 1, 128),
                MultimodalModality.stats: torch.randn(7, 10),
            }
        )


def test_multimodal_data_bundle_checks_target_consistency():
    with pytest.raises(ValueError, match="Target and modalities"):
        MultimodalDataBundle(
            modalities={
                MultimodalModality.raw: torch.randn(8, 1, 128),
            },
            target=torch.randint(0, 2, size=(7,)),
        )
