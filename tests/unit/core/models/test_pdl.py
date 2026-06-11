import pytest
import numpy as np
import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.models.pdl.pairwise_core import _predict_same_probability
from fedot_ind.core.models.pdl.pairwise_model import (
    PairwiseDifferenceEstimator,
    PairwiseDifferenceClassifier,
    PairwiseDifferenceRegressor,
)
from fedot_ind.core.models.pdl.pairwise_transform import PDCDataTransformer

# Fixtures for test data


@pytest.fixture
def classification_data():
    X, y = np.random.rand(50, 50), np.random.randint(0, 2, 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create InputData objects
    train_data = init_input_data(X=X_train, y=y_train, task='classification')
    test_data = init_input_data(X=X_test, y=y_test, task='classification')

    return train_data, test_data


@pytest.fixture
def regression_data():
    X, y = np.random.rand(50, 50), np.random.rand(50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create InputData objects
    train_data = init_input_data(X=X_train, y=y_train, task='regression')
    test_data = init_input_data(X=X_test, y=y_test, task='regression')

    return train_data, test_data

# Tests for PairwiseDifferenceEstimator


class TestPairwiseDifferenceEstimator:

    def test_convert_to_pandas(self):
        pde = PairwiseDifferenceEstimator()
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])

        df1, df2 = pde._convert_to_pandas(arr1, arr2)

        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert df1.shape == arr1.shape
        assert df2.shape == arr2.shape

    def test_pair_input(self):
        pde = PairwiseDifferenceEstimator()
        X1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        X2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        X_pair, X_pair_sym = pde.pair_input(X1, X2)

        # Test shapes
        assert X_pair.shape[0] == X1.shape[0] * X2.shape[0]  # Cross product
        assert X_pair_sym.shape[0] == X1.shape[0] * X2.shape[0]

        # Test column names
        expected_columns = ['a_x', 'b_x', 'a_y', 'b_y', 'a_diff', 'b_diff']
        assert all(col in X_pair.columns for col in expected_columns)

    def test_regression_pair_target_is_left_minus_anchor(self):
        pde = PairwiseDifferenceEstimator()
        y1 = pd.Series([1.0, 3.0])
        y2 = pd.Series([1.0, 3.0])

        delta = pde.pair_output(y1, y2)

        assert len(delta) == len(y1) * len(y2)
        # pairs: (1,1), (1,3), (3,1), (3,3) -> left - anchor
        np.testing.assert_allclose(delta, np.array([0.0, -2.0, 2.0, 0.0]))

    def test_classification_pair_target_semantics_same_is_zero_current_contract(self):
        pde = PairwiseDifferenceEstimator()
        y1 = pd.Series([5, 5, 7])
        y2 = pd.Series([5])

        dissimilarity_target = pde.pair_output_difference(y1, y2)

        # pairs: (5,5), (5,5), (7,5) -> same=0, same=0, different=1
        np.testing.assert_array_equal(dissimilarity_target, np.array([0, 0, 1]))

    def test_pair_output_difference(self):
        pde = PairwiseDifferenceEstimator()
        y1 = pd.Series([0, 1, 2])
        y2 = pd.Series([0, 2])

        y_pair_diff = pde.pair_output_difference(y1, y2, 3)

        # Test shape
        assert len(y_pair_diff) == len(y1) * len(y2)

        # Test values (should be 1 if different, 0 if same)
        expected = np.array([0, 1, 1, 1, 1, 0])  # Comparing [0,0], [0,2], [1,0], [1,2], [2,0], [2,2]
        np.testing.assert_array_equal(y_pair_diff, expected)


class _PairProbaStub:
    """заглушка sklearn-классификатора с predict_proba."""

    def __init__(self, probabilities: np.ndarray, classes: np.ndarray):
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, pair_features: np.ndarray) -> np.ndarray:
        return self._probabilities


class _HardLabelStub:
    """классификатора только с predict (без predict_proba)."""

    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def predict(self, pair_features: np.ndarray) -> np.ndarray:
        return self._labels


class TestPDLContracts:

    def test_predict_same_probability_uses_same_label_column(self):
        # predict_proba: столбец 0 = P(same), столбец 1 = P(different)
        stub_model = _PairProbaStub(
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
            classes=np.array([0, 1]),
        )

        same_probability = _predict_same_probability(stub_model, np.zeros((2, 3)))

        # Должны взять первый столбец (label 0), а не второй (label 1)
        np.testing.assert_allclose(same_probability, np.array([0.8, 0.3]))

    def test_predict_same_probability_falls_back_to_hard_predictions(self):
        # Без predict_proba: label 0 -> 1.0 (same), label 1 -> 0.0 (different)
        stub_model = _HardLabelStub(labels=np.array([0, 1, 0]))

        same_probability = _predict_same_probability(stub_model, np.zeros((3, 3)))

        np.testing.assert_array_equal(same_probability, np.array([1.0, 0.0, 1.0]))


class TestPDLDiagnostics:

    def test_pair_target_semantics_is_reported_in_diagnostics_classifier(self, classification_data):
        train_data, _ = classification_data
        classifier = PairwiseDifferenceClassifier(OperationParameters(model='rf', n_estimators=10, max_pairs=10_000))
        classifier.fit(train_data)

        semantics = classifier.get_diagnostics()["pair_target_semantics"]

        assert semantics["task"] == "classification"
        assert semantics["same_label"] == 0
        assert semantics["different_label"] == 1
        assert semantics["target_type"] == "dissimilarity"
        assert semantics["inference_output"] == "same_probability"

    def test_pair_target_semantics_is_reported_in_diagnostics_regressor(self, regression_data):
        train_data, _ = regression_data
        regressor = PairwiseDifferenceRegressor(OperationParameters(model='treg', n_estimators=10, max_pairs=10_000))
        regressor.fit(train_data)

        semantics = regressor.get_diagnostics()["pair_target_semantics"]

        assert semantics["task"] == "regression"
        assert semantics["delta_sign"] == "left_minus_anchor"
        assert semantics["target_formula"] == "target_left - target_anchor"
        assert semantics["inference_reconstruction"] == "anchor_target + predicted_delta"


class TestPDCDataTransformerContracts:

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame(
            {
                "numeric_col": [1.5, 2.7, 3.2],
                "string_col": ["A", "B", "C"],
            }
        )

    @pytest.mark.xfail(reason="PDCDataTransformer.fit() does not initialize preprocessing_ yet (Issue 2).")
    def test_pdc_data_transformer_initializes_x_preprocessor(self, sample_dataframe):
        transformer = PDCDataTransformer(y_type="numeric")
        transformer.fit(sample_dataframe, pd.Series([1.0, 2.0, 3.0]))

        assert hasattr(transformer, "preprocessing_")
        assert transformer.preprocessing_ is not None
        transformed = transformer.transform(sample_dataframe)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(sample_dataframe)

    @pytest.mark.xfail(reason="PDCDataTransformer.transform() should use preprocessing_y_ for y (Issue 2).")
    def test_pdc_data_transformer_uses_y_preprocessor_for_target(self, sample_dataframe):
        y = pd.Series([1.0, 2.0, 3.0], name="target")
        transformer = PDCDataTransformer(y_type="numeric")
        transformer.fit(sample_dataframe, y)

        assert isinstance(transformer.preprocessing_y_, StandardScaler)

        with patch.object(transformer.preprocessing_y_, "transform", wraps=transformer.preprocessing_y_.transform) as y_transform:
            with patch.object(transformer, "preprocessing_", create=True) as x_preprocessor:
                x_preprocessor.transform.return_value = sample_dataframe.values
                transformer.transform(sample_dataframe, y)

        y_transform.assert_called_once()


# Tests for PairwiseDifferenceClassifier


class TestPairwiseDifferenceClassifier:

    def test_init(self):
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        assert classifier.base_model_params == {'n_estimators': 10}
        assert hasattr(classifier, 'base_model')
        assert hasattr(classifier, 'pde')

    def test_fit_predict(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        # Test fit
        fitted_classifier = classifier.fit(train_data)
        assert fitted_classifier is classifier
        assert hasattr(classifier, 'num_classes')
        assert hasattr(classifier, 'target')
        assert hasattr(classifier, 'classes_')

        # Test predict
        predictions = classifier.predict(test_data, output_mode='labels')
        assert len(predictions) == len(test_data.target)

        # Test that predictions are valid class indices
        assert np.all(predictions >= 0)
        assert np.all(predictions < classifier.num_classes)

    def test_predict_proba(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        classifier.fit(train_data)

        # Test predict_proba
        proba = classifier.predict_proba(test_data, output_mode='default')

        # Check shape and probability sums
        assert proba.shape[0] == len(test_data.target)
        assert np.allclose(np.sum(proba, axis=1), np.ones(len(test_data.target)), atol=1e-10)

    def test_score_difference(self, classification_data):
        train_data, test_data = classification_data
        params = OperationParameters(model='rf', n_estimators=10)
        classifier = PairwiseDifferenceClassifier(params)

        classifier.fit(train_data)

        # Test score_difference
        score = classifier.score_difference(test_data)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # MAE value should be between 0 and 1

# Tests for PairwiseDifferenceRegressor


class TestPairwiseDifferenceRegressor:

    def test_init(self):
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        assert regressor.base_model_params == {'n_estimators': 10}
        assert hasattr(regressor, 'base_model')
        assert hasattr(regressor, 'pde')

    def test_fit_predict(self, regression_data):
        train_data, test_data = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        # Test fit
        fitted_regressor = regressor.fit(train_data)
        assert fitted_regressor is regressor
        assert hasattr(regressor, 'num_classes')
        assert hasattr(regressor, 'target')

        # Test predict
        predictions = regressor.predict(test_data)
        assert len(predictions) == len(test_data.target)

    def test_predict_samples(self, regression_data):
        train_data, test_data = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        regressor.fit(train_data)

        # Test _predict_samples
        prediction_samples, pred_diff_samples = regressor._predict_samples(test_data)

        # Check shapes
        assert prediction_samples.shape == (len(test_data.features), len(train_data.features))
        assert pred_diff_samples.shape == (len(test_data.features), len(train_data.features))

    # Skip testing learn_anchor_weights due to complexity
    # This would require mocking _name_to_method_mapping and detailed implementation knowledge

    def test_set_sample_weight(self, regression_data):
        train_data, _ = regression_data
        params = OperationParameters(model='treg', n_estimators=10)
        regressor = PairwiseDifferenceRegressor(params)

        regressor.fit(train_data)

        # Create a mock attribute for testing
        regressor.y_train_ = pd.Series(train_data.target, index=pd.RangeIndex(len(train_data.target)))

        # Test setting valid sample weights
        weights = pd.Series([1] * len(train_data.target), index=regressor.y_train_.index)
        regressor.set_sample_weight(weights)
        assert regressor.sample_weight_ is weights

        # Test with invalid weights (should raise error)
        with pytest.raises(ValueError):
            invalid_weights = pd.Series([1, 2])  # Wrong length
            regressor.set_sample_weight(invalid_weights)


