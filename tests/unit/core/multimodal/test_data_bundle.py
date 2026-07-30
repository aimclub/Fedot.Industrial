import pytest
import torch

from fedot_ind.core.multimodal.data_bundle import MultimodalDataBundle
from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.rules import ModalitySpec


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


def test_multimodal_data_bundle_requires_uniform_dtype():
    with pytest.raises(ValueError, match="same dtype"):
        MultimodalDataBundle(
            modalities={
                MultimodalModality.raw: torch.randn(2, 1, 4, dtype=torch.float32),
                MultimodalModality.stats: torch.randn(2, 3, dtype=torch.float64),
            }
        )


def test_multimodal_data_bundle_requires_uniform_device():
    with pytest.raises(ValueError, match="same device"):
        MultimodalDataBundle(
            modalities={
                MultimodalModality.raw: torch.ones(2, 1, 4),
                MultimodalModality.stats: torch.ones(2, 3, device="meta"),
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


def test_multimodal_data_bundle_rejects_derived_metadata_from_caller():
    with pytest.raises(ValueError, match="Derived metadata keys"):
        MultimodalDataBundle(
            modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
            metadata={"dtype": torch.float64},
        )


def test_multimodal_data_bundle_user_metadata_excludes_derived_keys():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        metadata={"source": {"split": "train"}},
    )

    assert bundle.user_metadata == {
        "source": {"split": "train"},
        "transform_params": {},
    }


def test_multimodal_data_bundle_is_immutable():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
    )

    with pytest.raises(AttributeError):
        bundle.target = torch.tensor([0, 1])

    with pytest.raises(TypeError):
        bundle.modalities[MultimodalModality.stats] = torch.randn(2, 3)

    with pytest.raises(TypeError):
        bundle.metadata["source"] = {}


def test_multimodal_data_bundle_copies_input_mapping():
    modalities = {MultimodalModality.raw: torch.randn(2, 1, 4)}
    bundle = MultimodalDataBundle(modalities=modalities)

    modalities[MultimodalModality.stats] = torch.randn(99, 3)

    assert bundle.available_modalities == [MultimodalModality.raw]


def test_multimodal_data_bundle_enforces_optional_rank_requirement():
    specs = {MultimodalModality.gaf: ModalitySpec(allowed_ndim=(4,))}

    with pytest.raises(ValueError, match="must have rank"):
        MultimodalDataBundle(
            modalities={MultimodalModality.gaf: torch.randn(2, 16, 16)},
            specs=specs,
        )

    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.gaf: torch.randn(2, 1, 16, 16)},
        specs=specs,
    )
    assert bundle.n_samples == 2


def test_multimodal_data_bundle_enforces_optional_shape_requirement():
    specs = {MultimodalModality.stats: ModalitySpec(allowed_ndim=(2,), shape=(10,))}

    with pytest.raises(ValueError, match="must have shape"):
        MultimodalDataBundle(
            modalities={MultimodalModality.stats: torch.randn(2, 9)},
            specs=specs,
        )


def test_multimodal_data_bundle_ignores_specs_for_absent_modalities():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        specs={MultimodalModality.gaf: ModalitySpec(allowed_ndim=(4,))},
    )

    assert bundle.available_modalities == [MultimodalModality.raw]


def test_modality_spec_rejects_shape_that_contradicts_rank():
    with pytest.raises(ValueError, match="allowed_ndim must be"):
        ModalitySpec(allowed_ndim=(2, 3), shape=(10,))

    with pytest.raises(ValueError, match="at least one rank"):
        ModalitySpec(allowed_ndim=())


def test_multimodal_data_bundle_replace_rebuilds_derived_metadata():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        target=torch.tensor([0, 1]),
        metadata={"source": {"split": "train"}},
    )

    enriched = bundle.with_metadata(source={"split": "test"})
    replaced = enriched.replace(
        modalities={MultimodalModality.raw: torch.randn(3, 1, 6)},
        target=None,
    )
    replaced_target = bundle.with_target(torch.tensor([1, 0]))

    assert enriched.metadata["source"] == {"split": "test"}
    assert replaced.target is None
    assert replaced.metadata["source"] == {"split": "test"}
    assert replaced.metadata["shapes"][MultimodalModality.raw] == (3, 1, 6)
    assert replaced_target.target.tolist() == [1, 0]


def test_multimodal_data_bundle_replace_keeps_target_when_omitted():
    target = torch.tensor([0, 1])
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        target=target,
    )

    assert bundle.replace(modalities=dict(bundle.modalities)).target is target
    assert bundle.without_target().target is None


def test_multimodal_data_bundle_rebuild_revalidates():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        target=torch.tensor([0, 1]),
    )

    with pytest.raises(ValueError, match="same number of samples"):
        bundle.with_modalities(
            {
                **bundle.modalities,
                MultimodalModality.stats: torch.randn(3, 6),
            }
        )

    with pytest.raises(ValueError, match="Target and modalities"):
        bundle.with_target(torch.tensor([0, 1, 2]))


def test_multimodal_data_bundle_to_updates_derived_metadata():
    bundle = MultimodalDataBundle(
        modalities={MultimodalModality.raw: torch.randn(2, 1, 4)},
        target=torch.tensor([0, 1]),
    )

    converted = bundle.to(dtype=torch.float64)

    assert converted.metadata["dtype"] == torch.float64
    assert converted.dtype == torch.float64
    assert converted.target.dtype == torch.int64
    assert bundle.metadata["dtype"] == torch.float32
    assert bundle.to() is bundle
