import numpy as np

from fedot_ind.core.kernel_learning import SummaryFeatureGenerator


def test_summary_feature_generator_is_deterministic_and_target_free():
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y_left = np.array([0, 1])
    y_right = np.array([1, 0])

    left = SummaryFeatureGenerator().fit_transform(X, y_left).features
    right = SummaryFeatureGenerator().fit_transform(X, y_right).features

    assert left.shape == (2, 10)
    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))
