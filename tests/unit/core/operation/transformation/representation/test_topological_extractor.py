import numpy as np
import pytest
from unittest.mock import patch
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.kernel_learning import create_feature_generator
from fedot_ind.core.operation.transformation.representation.topological.topological_extractor import TopologicalExtractor


@pytest.mark.parametrize("invalid_params, expected_error", [
    ({"stride": 0}, "Stride must be an integer >= 1"),
    ({"delay": -1}, "Delay must be an integer >= 1"),
    ({"multivariate_strategy": "concat"}, "Unsupported multivariate_strategy"),
    ({"filtration_type": "magic"}, "Unsupported filtration_type"),
    ({"backend": "numpy"}, "Unsupported backend"),
    ({"max_homology_dimension": -2}, "max_homology_dimension must be a non-negative integer"),
])
def test_topological_extractor_incorrect_params_raise_value_error(invalid_params, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        TopologicalExtractor(invalid_params)

def test_topological_extractor_ripser_alpha_warning():
    with pytest.warns(UserWarning, match="Methodology mismatch"):
        TopologicalExtractor({'backend': 'ripser++', 'filtration_type': 'alpha'})


def test_topological_extractor_input_dimensionality_warnings():
    pytest.importorskip("torch")
    extractor = TopologicalExtractor()
    X_2d = np.random.randn(2, 10) 
    
    with pytest.warns(UserWarning, match="Expected input tensor of shape"):
        features = extractor.generate_features_from_ts(X_2d)
        
    assert features is not None
    assert not features.empty

def test_topological_extractor_invalid_dimensionality_raises_error():
    pytest.importorskip("torch")
    extractor = TopologicalExtractor()
    X_4d = np.random.randn(2, 2, 2, 10)
    
    with pytest.raises(ValueError, match="Expected <= 3 dimensions"):
        extractor.generate_features_from_ts(X_4d)


def test_topological_extractor_multivariate_strategy_shapes():
    pytest.importorskip("torch")
    X = np.random.randn(2, 3, 15) 
    
    ext_joint = TopologicalExtractor({'multivariate_strategy': 'joint'})
    feat_joint = ext_joint.generate_features_from_ts(X)
    
    ext_ind = TopologicalExtractor({'multivariate_strategy': 'independent'})
    feat_ind = ext_ind.generate_features_from_ts(X)
    
    assert feat_joint.shape[0] == 2
    assert feat_ind.shape[0] == 2
    assert feat_ind.shape[1] > feat_joint.shape[1]
    assert feat_ind.shape[1] == feat_joint.shape[1] * 3

@patch('gtda.homology.VietorisRipsPersistence.fit_transform')
def test_topological_extractor_no_inf_or_nan_in_output(mock_vr):
    pytest.importorskip("torch")
    mock_diagram = np.array([[[0.0, np.inf, 0], [0.1, 0.5, 1]]])
    mock_vr.return_value = mock_diagram
    
    extractor = TopologicalExtractor()
    X = np.random.randn(1, 1, 15)
    features = extractor.generate_features_from_ts(X)
    
    assert np.isfinite(features.values).all()


def test_topological_extractor_uses_default_registry_path_for_small_input():
    pytest.importorskip("fedot")
    pytest.importorskip("torch")
    
    generator = create_feature_generator("topological_extractor")
    X = np.random.randn(2, 1, 15)
    y = np.array([0, 1])
    
    bundle = generator.fit_transform(X, y)
    
    is_skipped = bundle.diagnostics.get("budget", {}).get("skipped")
    skip_reason = bundle.diagnostics.get("budget", {}).get("skip_reason")
    
    assert is_skipped is not True, f"Fallback triggered! Reason: {skip_reason}"
    assert bundle.diagnostics.get("source") == "fedot_industrial_operation"