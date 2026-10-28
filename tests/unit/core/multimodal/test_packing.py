"""Schema, conservation, and storage-isolation laws for packed transport."""

from dataclasses import FrozenInstanceError
from math import prod

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from fedot_ind.core.multimodal.enums import MultimodalModality
from fedot_ind.core.multimodal.packing import (
    ModalityLayout,
    pack_modalities,
    parse_layout,
    unpack_modalities,
)


LAYOUT = [{"name": "raw", "shape": [1, 16]}, {"name": "stats", "shape": [3]}]


@st.composite
def valid_layouts(draw):
    names = draw(st.lists(st.sampled_from(list(MultimodalModality)), min_size=1, unique=True))
    result = []
    for name in names:
        rank = 2 if name is MultimodalModality.raw else 1 if name is MultimodalModality.stats else 3
        shape = draw(st.lists(st.integers(1, 4), min_size=rank, max_size=rank))
        result.append({"name": name.value, "shape": shape})
    return result


@given(payload=valid_layouts(), n_samples=st.integers(1, 5))
@settings(max_examples=40, deadline=None, database=None)
def test_roundtrip_preserves_named_values_shapes_and_schema_order(payload, n_samples):
    layout = parse_layout(payload)
    source = {
        item.name: (torch.arange(n_samples * prod(item.shape), dtype=torch.float64) + 1000 * i)
        .reshape(n_samples, *item.shape)
        for i, item in enumerate(layout)
    }
    reversed_source = dict(reversed(list(source.items())))
    packed = pack_modalities(reversed_source, layout)
    assert packed.shape == (n_samples, sum(item.width for item in layout))
    assert packed.dtype == torch.float64
    assert parse_layout(layout) == layout
    # Compare to an independent column oracle, not only the inverse operation.
    expected = torch.cat([source[item.name].flatten(start_dim=1) for item in layout], dim=1)
    assert torch.equal(packed, expected)
    restored = unpack_modalities(packed, layout)
    assert list(restored.modalities) == [MultimodalModality(item.name) for item in layout]
    for item in layout:
        assert torch.equal(restored.modalities[MultimodalModality(item.name)], source[item.name].float())


def test_layout_is_frozen_and_copies_shape():
    shape = [1, 16]
    layout = ModalityLayout("raw", shape)
    shape[1] = 99
    assert layout.shape == (1, 16)
    assert layout.width == 16
    with pytest.raises(FrozenInstanceError):
        layout.name = "stats"


@pytest.mark.parametrize("payload", [
    [],
    [{"name": "", "shape": [1]}],
    [{"name": "unknown", "shape": [1]}],
    [{"name": "raw", "shape": [1, True]}],
    [{"name": "raw", "shape": [1, 0]}],
    [{"name": "raw", "shape": [1, -1]}],
    [{"name": "raw", "shape": [1, 2.5]}],
    [{"name": "raw", "shape": [1]}],
    [{"name": "stats", "shape": [1, 2]}],
    [{"name": "raw", "shape": [[1], [16]]}],
    [{"name": "raw"}],
    [{"name": "raw", "shape": [1, 16], "extra": 1}],
    LAYOUT + [LAYOUT[0]],
])
def test_layout_rejects_invalid_records(payload):
    with pytest.raises(ValueError):
        parse_layout(payload)


@pytest.mark.parametrize("payload", [
    None, "[]", {}, [None], [[1, 16], [3]],
    [{"name": 3, "shape": [1]}],
    [{"name": "raw", "shape": "1,16"}],
])
def test_layout_rejects_wrong_payload_types(payload):
    with pytest.raises(TypeError):
        parse_layout(payload)


@pytest.mark.parametrize("numpy_input", [False, True])
def test_unpack_copies_input_modalities_and_target(numpy_input):
    packed = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    target = torch.tensor([4, 9])
    values = packed.numpy() if numpy_input else packed
    labels = target.numpy() if numpy_input else target
    bundle = unpack_modalities(values, LAYOUT, target=labels)
    bundle.modalities[MultimodalModality.raw].fill_(-1)
    bundle.modalities[MultimodalModality.stats].fill_(-2)
    bundle.target.fill_(0)
    assert torch.equal(packed, torch.arange(38, dtype=torch.float32).reshape(2, 19))
    assert torch.equal(target, torch.tensor([4, 9]))


def test_pack_accepts_enum_keys_noncontiguous_tensors_and_owns_storage():
    raw = torch.arange(32, dtype=torch.float64).reshape(2, 16, 1).transpose(1, 2)
    stats = torch.arange(6, dtype=torch.float64).reshape(3, 2).t()
    packed = pack_modalities({MultimodalModality.stats: stats, "raw": raw}, LAYOUT)
    packed.fill_(0)
    assert raw.sum() == 496
    assert stats.sum() == 15


@pytest.mark.parametrize("modalities,match", [
    ({"raw": torch.zeros(2, 1, 16)}, "missing"),
    ({"raw": torch.zeros(2, 1, 16), "stats": torch.zeros(2, 3), "other": torch.zeros(2, 1)}, "unknown"),
    ({"raw": torch.zeros(2, 2, 8), "stats": torch.zeros(2, 3)}, "shape"),
    ({"raw": torch.zeros(2, 1, 16), "stats": torch.zeros(3, 3)}, "same number"),
    ({"raw": torch.zeros(2, 1, 16), "stats": torch.zeros(2, 3, dtype=torch.float64)}, "same dtype"),
    ({"raw": torch.zeros(2, 1, 16), MultimodalModality.raw: torch.zeros(2, 1, 16),
      "stats": torch.zeros(2, 3)}, "Duplicate"),
])
def test_pack_rejects_missing_unknown_misaligned_or_lossy_modalities(modalities, match):
    with pytest.raises(ValueError, match=match):
        pack_modalities(modalities, LAYOUT)


@pytest.mark.parametrize("values,match", [
    (np.zeros(19), "2-D"),
    (torch.zeros(2, 1, 19), "2-D"),
    (np.zeros((2, 20)), "width"),
    (np.zeros((0, 19)), "at least one sample"),
    (np.full((2, 19), np.nan), "finite"),
    (torch.full((2, 19), float("inf")), "finite"),
    (torch.full((2, 19), 1e40, dtype=torch.float64), "float32"),
])
def test_unpack_rejects_invalid_matrix(values, match):
    with pytest.raises(ValueError, match=match):
        unpack_modalities(values, LAYOUT)


@pytest.mark.parametrize("values", [np.ones((2, 19), dtype=bool), np.ones((2, 19), dtype=complex),
                                    np.full((2, 19), "1", dtype=object), torch.ones(2, 19, dtype=torch.bool),
                                    [[1] * 19] * 2])
def test_unpack_rejects_non_real_numeric_matrix(values):
    with pytest.raises(TypeError):
        unpack_modalities(values, LAYOUT)


def test_unpack_negative_stride_numpy_and_target_shape_validation():
    values = np.arange(38).reshape(2, 19)[:, ::-1]
    bundle = unpack_modalities(values, LAYOUT)
    assert bundle.dtype == torch.float32
    assert bundle.modalities[MultimodalModality.raw][0, 0, 0] == 18
    with pytest.raises(ValueError, match="Target and modalities"):
        unpack_modalities(values, LAYOUT, target=torch.tensor([0]))


@pytest.mark.parametrize("storage", ["sparse", "meta"])
def test_pack_rejects_unsupported_tensor_storage_with_clear_error(storage):
    raw = torch.zeros(2, 1, 16).to_sparse() if storage == "sparse" else torch.empty(2, 1, 16, device="meta")
    with pytest.raises(TypeError, match="materialized strided"):
        pack_modalities({"raw": raw, "stats": torch.zeros(2, 3)}, LAYOUT)


def test_pack_rejects_non_tensor_and_nonfinite_values():
    with pytest.raises(TypeError, match="torch.Tensor"):
        pack_modalities({"raw": np.zeros((2, 1, 16)), "stats": torch.zeros(2, 3)}, LAYOUT)
    with pytest.raises(ValueError, match="finite"):
        pack_modalities({"raw": torch.full((2, 1, 16), float("nan")), "stats": torch.zeros(2, 3)}, LAYOUT)
