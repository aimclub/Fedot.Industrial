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
    assert "normalization" not in bundle.metadata
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


def test_multimodal_data_bundle_rejects_invalid_modalities():
    with pytest.raises(ValueError, match="at least one modality"):
        MultimodalDataBundle(modalities={})

    with pytest.raises(TypeError, match="Modality name"):
        MultimodalDataBundle(modalities={"raw": torch.randn(2, 1, 4)})

    with pytest.raises(TypeError, match="torch.Tensor"):
        MultimodalDataBundle(modalities={MultimodalModality.raw: [[1.0, 2.0]]})

    with pytest.raises(ValueError, match="sample dimension"):
        MultimodalDataBundle(modalities={MultimodalModality.raw: torch.tensor(1.0)})


def test_multimodal_data_bundle_checks_target_consistency():
    with pytest.raises(ValueError, match="Target and modalities"):
        MultimodalDataBundle(
            modalities={
                MultimodalModality.raw: torch.randn(8, 1, 128),
            },
            target=torch.randint(0, 2, size=(7,)),
        )


def test_multimodal_data_bundle_checks_target_type_and_rank():
    with pytest.raises(TypeError, match="Target must be torch.Tensor"):
        MultimodalDataBundle(
            modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
            target=[0, 1],
        )

    with pytest.raises(ValueError, match="sample dimension"):
        MultimodalDataBundle(
            modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
            target=torch.tensor(1),
        )


def test_multimodal_data_bundle_replace_rebuilds_auto_metadata():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        target=torch.tensor([0, 1]),
        metadata={"source": {"split": "train"}},
    )

    enriched = bundle.with_metadata(source={"split": "test"})
    replaced = enriched.replace(
        modalities={MultimodalModality.raw: torch.randn(3, 1, 6)},
        keep_target=False,
    )
    replaced_target = bundle.replace(target=torch.tensor([1, 0]))

    assert enriched.metadata["source"] == {"split": "test"}
    assert replaced.target is None
    assert replaced.metadata["source"] == {"split": "test"}
    assert replaced.metadata["shapes"][MultimodalModality.raw] == (3, 1, 6)
    assert replaced_target.target.tolist() == [1, 0]
