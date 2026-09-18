import numpy as np

from benchmark.industrial.core import ModelSpec
from benchmark.industrial.models.classification import build_classification_model


def test_sklearn_classifier_adapter_can_wrap_feature_generators():
    """The benchmark adapter must preserve the native Riemann lifecycle."""
    train_features = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [1.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        ],
        dtype=float,
    )
    train_target = np.array([0, 0, 1, 1, 0, 1], dtype=object)
    test_features = np.array(
        [
            [0.2, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6],
            [8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5],
        ],
        dtype=float,
    )

    adapter = build_classification_model(
        ModelSpec(
            adapter_name='sklearn_classifier',
            display_name='Riemann+Logit',
            params={
                'generator_name': 'riemann_extractor',
                'generator_params': {
                    'feature_mode': 'mdm',
                    'mdm_centroid_scope': 'global',
                },
                'classifier_name': 'logistic_regression',
                'classifier_params': {'max_iter': 5000},
            },
        )
    )

    adapter.fit(train_features, train_target)
    prediction = adapter.predict(test_features)

    assert adapter.generator_.fallback_generator_ is None
    assert adapter.generator_.operation_specs[0].fit_transform_on_fit is True
    assert prediction.shape == (2,)
    assert set(prediction) <= {'0', '1'}
