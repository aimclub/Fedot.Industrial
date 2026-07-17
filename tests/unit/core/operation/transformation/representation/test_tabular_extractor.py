import numpy as np

from fedot_ind.core.operation.transformation.representation.tabular.tabular_extractor import TabularExtractor


def test_tabular_reducer_pads_short_transform_features_to_fitted_width():
    extractor = TabularExtractor({'reduce_dimension': True})
    train_features = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.1],
        [1.0, 1.0, 0.0, 0.3],
    ])
    train_target = np.array([0, 1, 0, 1])

    train_reduced = extractor._reduce_dim(train_features, train_target)
    test_reduced = extractor._reduce_dim(np.array([[1.0, 0.0], [0.0, 1.0]]), None)

    assert test_reduced.shape[0] == 2
    assert test_reduced.shape[1] == train_reduced.shape[1]
    assert extractor.feature_alignment_ == {
        'expected_width': 4,
        'actual_width': 2,
        'action': 'padded',
    }


def test_tabular_reducer_truncates_wide_transform_features_to_fitted_width():
    extractor = TabularExtractor({'reduce_dimension': True})
    train_features = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    train_reduced = extractor._reduce_dim(train_features, np.array([0, 1, 0, 1]))
    test_reduced = extractor._reduce_dim(np.array([[1.0, 0.0, 0.0, 3.0, 4.0]]), None)

    assert test_reduced.shape[0] == 1
    assert test_reduced.shape[1] == train_reduced.shape[1]
    assert extractor.feature_alignment_ == {
        'expected_width': 3,
        'actual_width': 5,
        'action': 'truncated',
    }
