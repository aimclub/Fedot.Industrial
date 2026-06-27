import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pytest
from fedot_ind.core.kernel_learning import (
    BudgetedRepositoryFeatureGeneratorAdapter,
    GeneratorBudgetPolicy,
    OperationSpec,
    RepositoryFeatureGeneratorAdapter,
    ShapeletFeatureGenerator,
    SummaryFeatureGenerator,
    build_generator_registry,
    create_feature_generator,
    resolve_torch_device,
)
from fedot_ind.core.kernel_learning.generators import adapters
from fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding import RiemannExtractor

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@dataclass
class MockInputData:
    features: np.ndarray
    target: Optional[np.ndarray] = None


def generate_spd_matrices(n_samples: int, channels: int) -> np.ndarray:
    A = np.random.randn(n_samples, channels, channels)
    return A @ A.transpose(0, 2, 1) + np.eye(channels) * 1e-4

@pytest.fixture
def spd_data():
    X = generate_spd_matrices(10, 3)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return InputData(
        idx=np.arange(10),
        features=X,
        target=y,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table
    )


def test_riemann_extractor_handles_short_series_without_nan_or_inf():
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    features = generator.fit_transform(X).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_riemann_extractor_sanitizes_nan_and_inf_inputs():
    generator = create_feature_generator("riemann_extractor")
    X = np.array(
        [
            [0.0, np.nan, 1.0],
            [np.inf, -1.0, 0.0],
            [1.0, 0.0, 0.5],
        ]
    )

    features = generator.fit_transform(X).features

    assert features.shape[0] == X.shape[0]
    assert np.all(np.isfinite(features))


def test_riemann_extractor_fit_transform_and_transform_are_target_free():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y_left = np.array([0, 1])
    y_right = np.array([1, 0])

    generator_left = create_feature_generator("riemann_extractor")
    generator_right = create_feature_generator("riemann_extractor")

    left = generator_left.fit_transform(X, y_left).features
    right = generator_right.fit_transform(X, y_right).features

    assert np.allclose(left, right)
    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
    assert left.shape == right.shape


def test_riemann_extractor_same_for_classification_and_regression_and_ts_forecasting():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")

    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    y = np.array([0, 1])

    gen_clf = create_feature_generator("riemann_extractor")
    out_clf = gen_clf.fit_transform(X, y, task_type="classification").features

    gen_reg = create_feature_generator("riemann_extractor")
    out_reg = gen_reg.fit_transform(X, y, task_type="regression").features

    gen_ts = create_feature_generator("riemann_extractor")
    out_ts = gen_ts.fit_transform(X, y, task_type="ts_forecasting").features

    assert out_clf.shape == out_reg.shape == out_ts.shape
    assert np.all(np.isfinite(out_clf))
    assert np.all(np.isfinite(out_reg))
    assert np.all(np.isfinite(out_ts))
    assert np.allclose(out_clf, out_reg)
    assert np.allclose(out_clf, out_ts)


@pytest.mark.parametrize("invalid_params, expected_error_match", [
    (
        {"extraction_strategy": "magic_method"}, 
        "Unsupported extraction strategy: 'magic_method'"
    ),
    (
        {"estimator": "pearson"}, 
        "Unsupported estimator: 'pearson'"
    ),
    (
        {"SPD_metric": "manhattan"}, 
        "Unsupported SPD_metric: 'manhattan'"
    ),
    (
        {"tangent_metric": "cosine"}, 
        "Unsupported tangent_metric: 'cosine'"
    ),
])
def test_riemann_extractor_incorrect_params_raise_value_error(invalid_params, expected_error_match):

    with pytest.raises(ValueError, match=expected_error_match):
        RiemannExtractor(invalid_params)


def test_riemann_extractor_standalone_mdm_strategy_fit_without_target_raises_error(spd_data):
    extractor = RiemannExtractor({
        'extraction_strategy': 'mdm',
        'centroid_strategy': 'class-wise'
    })
    
    data_without_target = MockInputData(features=spd_data.features, target=None)
    
    with pytest.raises(ValueError, match="Target data is required to fit MDM centroids"):
        extractor.fit(data_without_target)


def test_riemann_extractor_centroid_strategy_behavioral_difference(spd_data):
    ext_class_wise = RiemannExtractor({
        'extraction_strategy': 'mdm',
        'centroid_strategy': 'class-wise'
    }).fit(spd_data)
    
    ext_global = RiemannExtractor({
        'extraction_strategy': 'mdm',
        'centroid_strategy': 'global'
    }).fit(spd_data)

    assert len(ext_class_wise.covmeans_) == 2
    assert len(ext_global.covmeans_) == 1

    out_class_wise = ext_class_wise._transform(spd_data)
    out_global = ext_global._transform(spd_data)

    assert out_class_wise.shape[1] == 2
    assert out_global.shape[1] == 1
    assert not np.allclose(out_class_wise[:, 0], out_global[:, 0])


@pytest.mark.parametrize("strategy, expected_dim", [
    ('mdm', 2),          
    ('tangent', 6),      
    ('ensemble', 8)      
])
def test_riemann_extractor_output_shape_matches_strategy(strategy, expected_dim, spd_data):
    extractor = RiemannExtractor({
        'extraction_strategy': strategy,
        'centroid_strategy': 'class-wise' if strategy != 'tangent' else 'global'
    })
    

    features = extractor.fit(spd_data)._transform(spd_data)
        
    assert features.shape == (10, expected_dim)


@patch('fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding.median_riemann')
def test_riemann_extractor_robust_centroid_dispatching(mock_median_riemann, spd_data):
    mock_median_riemann.return_value = np.eye(3)
    
    extractor = RiemannExtractor({
        'extraction_strategy': 'mdm',
        'centroid_strategy': 'global',
        'centroid_type': 'median',
        'SPD_metric': 'riemann'
    })
    extractor.fit(spd_data)
    
    mock_median_riemann.assert_called_once()


def test_riemann_extractor_median_unsupported_metric_raises_error():
    with pytest.raises(ValueError, match="only natively supported for 'riemann' and 'euclid'"):
        RiemannExtractor({
            'centroid_type': 'median',
            'SPD_metric': 'logeuclid'
        })


def test_riemann_extractor_tangent_with_class_wise_raises_warning():
    with pytest.warns(UserWarning, match="Conceptual mismatch"):
        RiemannExtractor({
            'extraction_strategy': 'tangent',
            'centroid_strategy': 'class-wise'
        })


def test_riemann_extractor_tangent_with_median_raises_warning():
    with pytest.warns(UserWarning, match="Methodology mismatch"):
        RiemannExtractor({
            'extraction_strategy': 'tangent',
            'centroid_type': 'median'
        })


def test_riemann_extractor_transform_before_fit_raises_warning(spd_data):
    extractor = RiemannExtractor({'extraction_strategy': 'mdm'})
    
    with pytest.warns(UserWarning, match="RiemannExtractor is not fitted"):
        extractor._transform(spd_data)


@patch('fedot_ind.core.operation.transformation.representation.manifold.riemann_embeding.distance')
def test_riemann_extractor_distance_uses_spd_metric(mock_distance, spd_data):
    mock_distance.return_value = np.ones(10)
    
    extractor = RiemannExtractor({
        'extraction_strategy': 'mdm',
        'centroid_strategy': 'global',
        'SPD_metric': 'logeuclid',
        'tangent_metric': 'riemann'
    })
    
    extractor.fit(spd_data)
    extractor._transform(spd_data)
    
    mock_distance.assert_called()
    call_kwargs = mock_distance.call_args[1]
    assert call_kwargs.get('metric') == 'logeuclid'