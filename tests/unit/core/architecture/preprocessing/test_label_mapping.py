"""Unit tests for contiguous class-label encoding."""

from __future__ import annotations

import numpy as np
import pytest

from fedot_ind.core.architecture.preprocessing.label_mapping import (
    ContiguousLabelEncoder,
    LabelMappingError,
)
from fedot_ind.core.multimodal import MultimodalDatasetPreparer, build_preparation_config


def test_contiguous_label_encoder_maps_gapped_integer_labels():
    encoder = ContiguousLabelEncoder()
    encoded = encoder.fit_transform([5, 0, 2, 5])

    assert encoded == [2, 0, 1, 2]
    assert encoder.num_classes == 3
    assert encoder.as_label_mapping() == {0: 0, 2: 1, 5: 2}
    assert encoder.inverse_transform([0, 1, 2]) == [0, 2, 5]


def test_contiguous_label_encoder_rejects_unknown_labels_and_oov_indices():
    encoder = ContiguousLabelEncoder().fit(['a', 'b'])

    with pytest.raises(LabelMappingError, match='Unknown target labels'):
        encoder.transform(['a', 'c'])

    with pytest.raises(LabelMappingError, match='out of vocabulary'):
        encoder.inverse_transform([0, 99])


def test_preparer_remaps_gapped_integer_targets_with_shared_encoder():
    config = build_preparation_config(
        transformation_config={'raw': {'per_sample_z_normalize': False}},
        normalization_config={},
    )
    preparer = MultimodalDatasetPreparer(config=config)
    bundle = preparer.fit_transform(
        np.zeros((3, 4)),
        np.asarray([0, 2, 5], dtype=np.int64),
    )

    assert bundle.target.tolist() == [0, 1, 2]
    assert preparer.label_mapping_ == {0: 0, 2: 1, 5: 2}
    assert preparer.label_encoder_.inverse_transform([0, 1, 2]) == [0, 2, 5]
