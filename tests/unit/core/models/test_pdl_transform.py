import importlib
import sys

import pytest
import numpy as np
import pandas as pd
# from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from unittest.mock import MagicMock, patch

from fedot_ind.core.models.pdl.legacy_pairwise_transform import PDCDataTransformer, SampleWeights
from fedot.core.operations.operation_parameters import OperationParameters


def test_pairwise_transform_module_is_deprecated_compatibility_layer():
    sys.modules.pop("fedot_ind.core.models.pdl.pairwise_transform", None)
    with pytest.warns(DeprecationWarning, match="pairwise_transform is deprecated"):
        legacy_module = importlib.import_module(
            "fedot_ind.core.models.pdl.pairwise_transform")

    assert legacy_module.PDCDataTransformer is PDCDataTransformer
    assert legacy_module.SampleWeights is SampleWeights


class TestPDCDataTransformer:
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with mixed types for testing."""
        data = {
            'numeric_col1': [1.5, 2.7, 3.2, 4.1, 5.0],
            'numeric_col2': [10, 20, 30, 40, 50],
            'ordinal_col': pd.Categorical(['low', 'medium', 'high', 'medium', 'low'], ordered=True),
            'string_col1': ['A', 'B', 'C', 'D', 'E'],
            'string_col2': pd.Categorical(['cat', 'dog', 'cat', 'bird', 'dog'], ordered=False),
            'bool_col': [True, False, True, False, True]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def y_numeric(self):
        """Sample numeric target variable."""
        return pd.Series([5.1, 6.2, 7.3, 8.4, 9.5], name='target')

    @pytest.fixture
    def y_categorical(self):
        """Sample categorical target variable."""
        return pd.Series(['X', 'Y', 'Z', 'X', 'Y'], name='target')

    def test_init_default(self):
        """Test initialization with default parameters."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer = PDCDataTransformer()
        assert transformer.numeric_features is None
        assert transformer.ordinal_features is None
        assert transformer.string_features is None
        assert transformer.y_type is None

    def test_init_with_params(self):
        """Test initialization with specified parameters."""
        numeric = ['num1', 'num2']
        ordinal = ['ord1']
        string = ['str1', 'str2']

        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer = PDCDataTransformer(
                numeric_features=numeric,
                ordinal_features=ordinal,
                string_features=string,
                y_type='numeric'
            )

        assert transformer.numeric_features == numeric
        assert transformer.ordinal_features == ordinal
        assert transformer.string_features == string
        assert transformer.y_type == 'numeric'

    def test_init_invalid_y_type(self):
        """Test initialization with invalid y_type parameter."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            with pytest.raises(ValueError) as excinfo:
                PDCDataTransformer(y_type='invalid_type')
        assert "y_type must be one of 'numeric', 'ordinal', 'string'" in str(
            excinfo.value)

    def test_fit_auto_feature_detection(self, sample_dataframe):
        """Test automatic feature type detection during fit."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer = PDCDataTransformer()
        transformer.fit(sample_dataframe)

        assert set(transformer.numeric_features) == {
            'numeric_col1', 'numeric_col2', 'bool_col'}

    def test_fit_y_transformers(self, sample_dataframe, y_numeric, y_categorical):
        """Test fitting of y preprocessors."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer_num = PDCDataTransformer(y_type='numeric')
            transformer_ord = PDCDataTransformer(y_type='ordinal')
            transformer_str = PDCDataTransformer(y_type='string')

        transformer_num.fit(sample_dataframe, y_numeric)
        assert isinstance(transformer_num.preprocessing_y_, StandardScaler)

        transformer_ord.fit(sample_dataframe, y_categorical)
        assert isinstance(transformer_ord.preprocessing_y_, OrdinalEncoder)

        transformer_str.fit(sample_dataframe, y_categorical)

        # Test string target
        # transformer_str = PDCDataTransformer(y_type='string')
        # assert isinstance(transformer_str.preprocessing_y_, OneHotEncoder)
        assert isinstance(transformer_str.preprocessing_y_, OneHotEncoder)

    def test_cast_uint(self, sample_dataframe, y_numeric):
        """Test the cast_uint method."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer = PDCDataTransformer()
        X_cast, y_cast = transformer.cast_uint(sample_dataframe, y_numeric)

        assert X_cast['numeric_col1'].dtype == np.float32
        assert X_cast['numeric_col2'].dtype == np.float32
        assert y_cast.dtype == np.float32

    def test_transform_raises_not_implemented(self, sample_dataframe):
        """Legacy transformer must not expose a broken production transform path."""
        with pytest.warns(DeprecationWarning, match="PDCDataTransformer"):
            transformer = PDCDataTransformer()
        transformer.fit(sample_dataframe)
        with pytest.raises(NotImplementedError, match="legacy module"):
            transformer.transform(sample_dataframe)


class TestSampleWeights:
    @pytest.fixture
    def sample_params(self):
        """Create sample parameters for SampleWeights."""
        params = OperationParameters(**{'method': 'L2'})
        return params

    @pytest.fixture
    def sample_weights_instance(self, sample_params):
        """Create SampleWeights instance with configured parameters."""
        with pytest.warns(DeprecationWarning, match="SampleWeights"):
            return SampleWeights(params=sample_params)

    @pytest.fixture
    def mock_training_data(self):
        """Create mock training data."""
        X_train = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        return X_train

    @pytest.fixture
    def mock_validation_data(self):
        """Create mock validation data."""
        X_val = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5],
            'feature2': [0.15, 0.25, 0.35]
        })
        y_val = pd.Series([10.0, 20.0, 30.0])
        return X_val, y_val

    def test_init(self, sample_params):
        """Test initialization with parameters."""
        with pytest.warns(DeprecationWarning, match="SampleWeights"):
            weights = SampleWeights(params=sample_params)
        assert weights.method == 'L2'
        assert 'L2' in weights.method_dict

    def test_init_default(self):
        """Test initialization with default parameters."""
        with pytest.warns(DeprecationWarning, match="SampleWeights"):
            weights = SampleWeights(dict())
        assert weights.method == 'L2'  # Default method

    def test_normalize_weights(self, sample_weights_instance):
        """Test normalize_weights method."""
        # Test with varied weights
        weights = pd.Series([1.0, 2.0, 3.0, 4.0])
        normalized = sample_weights_instance._normalize_weights(weights)
        assert np.isclose(normalized.sum(), 1.0)
        assert all(normalized >= 0)

        # Test with uniform weights
        uniform_weights = pd.Series([2.0, 2.0, 2.0, 2.0])
        normalized_uniform = sample_weights_instance._normalize_weights(
            uniform_weights)
        assert np.isclose(normalized_uniform.sum(), 1.0)
        assert all(normalized_uniform == 0.25)

    def test_normalize_weights_negative(self, sample_weights_instance):
        """Test normalize_weights with negative values."""
        weights = pd.Series([1.0, -2.0, 3.0, 4.0])
        with pytest.raises(AssertionError):
            sample_weights_instance._normalize_weights(weights)

    @patch.object(SampleWeights, '_sample_weight_by_kmeans_prototypes')
    def test_method_kmeans(self, mock_kmeans):
        """Test KMeansClusterCenters method selection."""
        params = OperationParameters(**{'method': 'KMeansClusterCenters'})
        with pytest.warns(DeprecationWarning, match="SampleWeights"):
            weights = SampleWeights(params=params)

        # Configure mock
        expected_result = pd.Series([0.2, 0.3, 0.5])
        mock_kmeans.return_value = expected_result

        # Setup attributes needed for method execution
        weights.X_train_ = pd.DataFrame({'f1': [1, 2, 3]})

        # Test method call
        result = weights.method_dict['KMeansClusterCenters']()

        # Assertions
        mock_kmeans.assert_called_once()
        assert result.equals(expected_result)

    @patch.object(SampleWeights, '_sample_weight_optimize')
    def test_method_l2(self, mock_optimize):
        """Test L2 method selection and configuration."""
        params = OperationParameters(**{'method': 'L2'})
        with pytest.warns(DeprecationWarning, match="SampleWeights"):
            weights = SampleWeights(params=params)

        # Test partial function configuration
        partial_func = weights.method_dict['L2']
        assert partial_func.func == weights._sample_weight_optimize
        assert partial_func.keywords == {'l2_lambda': 0.1}

    # @patch('fedot_ind.core.models.pdl.pairwise_transform.minimize')
    @patch('fedot_ind.core.models.pdl.legacy_pairwise_transform.minimize')
    def test_sample_weight_optimize(
            self,
            mock_minimize,
            sample_weights_instance,
            mock_training_data,
            mock_validation_data):
        """Test sample_weight_optimize method."""
        # Setup
        X_val, y_val = mock_validation_data
        sample_weights_instance.X_train_ = mock_training_data

        # Configure the _predict_samples mock
        with patch.object(
            SampleWeights,
            '_predict_samples',
            return_value=(pd.DataFrame(np.random.rand(3, 5)), None),
            create=True,
        ):
            # Configure minimize mock
            mock_result = MagicMock()
            mock_result.x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            mock_minimize.return_value = mock_result

            # Test
            result = sample_weights_instance._sample_weight_optimize(
                X_val, y_val)

            # Assertions
            assert mock_minimize.called
            assert isinstance(result, pd.Series)
            assert len(result) == len(mock_training_data)
            assert np.isclose(result.sum(), 1.0)

    def test_sample_weight_ordered_votes_from_weights(self, sample_weights_instance):
        """Test _sample_weight_ordered_votes_from_weights static method."""
        received_weights = np.array([0.1, 0.3, 0.2, 0.4])
        result = SampleWeights._sample_weight_ordered_votes_from_weights(
            received_weights)

        # Highest received weight gets the largest rank-derived vote.
        expected = np.array([0.1, 0.3, 0.2, 0.4])
        assert np.allclose(result, expected)

    @patch.object(SampleWeights, '_sample_weight_negative_error')
    def test_sample_weight_ordered_votes(self, mock_neg_error, sample_weights_instance, mock_validation_data):
        """Test _sample_weight_ordered_votes method."""
        X_val, y_val = mock_validation_data

        # Configure mock
        mock_neg_error.return_value = pd.Series([0.1, 0.3, 0.2, 0.4, 0.0])

        # Mock the static method
        with patch.object(
            SampleWeights,
            '_sample_weight_ordered_votes_from_weights',
            return_value=np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        ) as mock_ordered:
            # Test
            result = sample_weights_instance._sample_weight_ordered_votes(
                X_val, y_val)

            # Assertions
            mock_neg_error.assert_called_once_with(
                X_val, y_val, force_symmetry=True)
            mock_ordered.assert_called_once_with(mock_neg_error.return_value)
            assert np.array_equal(result, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

    def test_error_inverse_and_negative_error_weighting(self, sample_weights_instance):
        """Legacy weighting helpers normalize per-anchor validation errors."""
        sample_weights_instance.X_train_ = pd.DataFrame(index=[0, 1, 2])
        prediction_samples = pd.DataFrame(
            {
                0: [1.0, 2.0, 3.0],
                1: [1.0, 2.0, 4.0],
                2: [3.0, 3.0, 3.0],
            },
            index=[0, 1, 2],
        )
        y_val = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])

        with patch.object(
            sample_weights_instance,
            '_predict_samples',
            return_value=(prediction_samples, None),
            create=True,
        ):
            val_mae = sample_weights_instance._error(
                pd.DataFrame({"x": [0, 1, 2]}), y_val)
            inverse_weights = sample_weights_instance._sample_weight_inverse_error(
                pd.DataFrame({"x": [0, 1, 2]}), y_val)
            negative_weights = sample_weights_instance._sample_weight_negative_error(
                pd.DataFrame({"x": [0, 1, 2]}), y_val)

        np.testing.assert_allclose(
            val_mae.to_numpy(), np.array([0.0, 1.0 / 3.0, 1.0]))
        assert np.isclose(inverse_weights.sum(), 1.0)
        assert np.isclose(negative_weights.sum(), 1.0)
        assert inverse_weights.iloc[0] > inverse_weights.iloc[-1]

    def test_negative_error_returns_uniform_weights_for_zero_or_degenerate_errors(
            self,
            sample_weights_instance):
        sample_weights_instance.X_train_ = pd.DataFrame(index=[0, 1, 2])

        with patch.object(sample_weights_instance, '_error', return_value=pd.Series([0.0, 0.0, 0.0])):
            zero_weights = sample_weights_instance._sample_weight_negative_error(
                pd.DataFrame({"x": [0]}), pd.Series([0.0]))
        with patch.object(sample_weights_instance, '_error', return_value=pd.Series([1.0, 1.0, 1.0])):
            flat_weights = sample_weights_instance._sample_weight_negative_error(
                pd.DataFrame({"x": [0]}), pd.Series([0.0]))

        np.testing.assert_allclose(
            zero_weights.to_numpy(), np.repeat(1 / 3, 3))
        np.testing.assert_allclose(
            flat_weights.to_numpy(), np.repeat(1 / 3, 3))

    def test_sample_weight_extreme_pruning_retries_until_weights_are_not_sparse(
            self,
            sample_weights_instance):
        sparse = pd.Series([0.0] * 10 + [1.0])
        dense = pd.Series(np.repeat(1 / 11, 11))
        with patch.object(
            sample_weights_instance,
            '_sample_weight_optimize',
            side_effect=[sparse, dense],
        ) as mock_optimize:
            result = sample_weights_instance._sample_weight_extreme_pruning(
                pd.DataFrame({"x": [0]}), pd.Series([0.0]))

        assert mock_optimize.call_count == 2
        assert result.equals(dense)

    @patch('fedot_ind.core.models.pdl.legacy_pairwise_transform.cdist')
    @patch('fedot_ind.core.models.pdl.legacy_pairwise_transform.KMeans')
    def test_sample_weight_by_kmeans_prototypes_marks_closest_rows(
            self,
            mock_kmeans_cls,
            mock_cdist,
            sample_weights_instance):
        sample_weights_instance.X_train_ = pd.DataFrame(
            {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
            index=["a", "b", "c"],
        )
        mock_kmeans = MagicMock()
        mock_kmeans.cluster_centers_ = np.array([[0.0, 0.0], [2.0, 2.0]])
        mock_kmeans_cls.return_value = mock_kmeans
        mock_cdist.return_value = np.array(
            [
                [0.0, 3.0],
                [1.0, 1.0],
                [3.0, 0.0],
            ]
        )

        weights = sample_weights_instance._sample_weight_by_kmeans_prototypes(
            k=2)

        mock_kmeans.fit.assert_called_once_with(
            sample_weights_instance.X_train_)
        np.testing.assert_allclose(
            weights.to_numpy(), np.array([0.5, 0.0, 0.5]))
