from typing import Optional, List
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin, BaseEstimator
from sktime.classification.dictionary_based import WEASEL
from fedot_ind.core.architecture.settings.computational import backend_methods as np


class EarlyTSClassifier(ClassifierMixin, BaseEstimator):
    """
    Base class for Early Time Series Classification models 
    which implement prefix-wise predictions via traiing multiple slave estimators.

    Args:
        ``interval_percentage (float in (1, 100])``: define how much points should be between prediction points.
        ``consecutive_predictions (int)``: how many last subsequent estimators should classify object equally.
        ``accuracy_importance (float in [0, 1])``: trade-off coefficient between earliness and accuracy.
        ``prediction_mode (str in ['last_available', 'best_by_metrics_mean', 'all'])``:
            - if 'last_available', returns the latest estimator prediction allowed by prefix length;
            - if 'best_by_metrics_mean', returns the best of estimators estimated
              with weighted average of accuracy and earliness
            - if 'all', returns all estiamtors predictions
        ``transform_score (bool)``: whether or not to scale scores to [-1, 1] interval
        ``min_ts_step (int)``: minimal difference between to subsequent prefix' lengths
    """
    def __init__(self, params: Optional[OperationParameters] = {}):     
        super().__init__()
        self.interval_percentage = params.get('interval_percentage', 10)
        self.consecutive_predictions = params.get('consecutive_predictions', 1)
        self.accuracy_importance = params.get('accuracy_importance', 1.)
        self.min_ts_length = params.get('min_ts_step', 3)
        self.random_state = params.get('random_state', None)
        
        self.prediction_mode = params.get('prediction_mode', 'last_available')
        self.transform_score = params.get('transform_score', True)
        self.weasel_params = {}

    def _init_model(self, X, y):
        max_data_length = X.shape[-1]
        self.prediction_idx = self._compute_prediction_points(max_data_length)
        self.n_pred = len(self.prediction_idx)
        self.slave_estimators = [
            WEASEL(random_state=self.random_state, support_probabilities=True, **self.weasel_params)
            for _ in range(self.n_pred)]
        self.scalers = [StandardScaler() for _ in range(self.n_pred)]
        self._chosen_estimator_idx = -1
        self.classes_ = [np.unique(y)]
        self._estimator_for_predict = [-1]

    @property
    def required_length(self):
        if not hasattr(self, '_chosen_estimator_idx'):
            return None
        return self.prediction_idx[self._chosen_estimator_idx]

    @property
    def n_classes(self):
        return len(self.classes_[0])

    def fit(self, X, y=None):
        assert y is not None, 'Pass y'
        y = np.array(y).flatten()
        self._init_model(X, y)
        for i in range(self.n_pred):
            self._fit_one_interval(X, y, i)

    def _fit_one_interval(self, X, y, i):
        X_part = X[..., :self.prediction_idx[i] + 1]
        X_part = self.scalers[i].fit_transform(X_part)
        probas = self.slave_estimators[i].fit_predict_proba(X_part, y)
        return probas

    def _predict_one_slave(self, X, i, offset=0):
        X_part = X[..., max(0, offset - 1):self.prediction_idx[i] + 1]
        X_part = self.scalers[i].transform(X_part)
        probas = self.slave_estimators[i].predict_proba(X_part)
        return probas, np.argmax(probas, axis=-1)

    def _compute_prediction_points(self, n_idx):
        interval_length = max(int(n_idx * self.interval_percentage / 100), self.min_ts_length)
        prediction_idx = np.arange(n_idx - 1, -1, -interval_length)[::-1][1:]
        self.earliness = 1 - prediction_idx / n_idx  # /n_idx because else the last hm score is always 0
        return prediction_idx

    def _select_estimators(self, X, training=False):
        offset = 0
        if not training and self.prediction_mode == 'best_by_metrics_mean':
            estimator_indices = [self._chosen_estimator_idx]
        elif not training and self.prediction_mode == 'last_available':
            last_idx, offset = self._get_applicable_index(X.shape[-1] - 1)
            estimator_indices = [last_idx]
        elif training or self.prediction_mode == 'all':
            last_idx, offset = self._get_applicable_index(X.shape[-1] - 1)
            estimator_indices = np.arange(last_idx + 1)
        else:
            raise ValueError('Unknown prediction mode')
        return estimator_indices, offset

    def _predict(self, X, training=True):
        estimator_indices, offset = self._select_estimators(X, training)
        if not training:
            self._estimator_for_predict = estimator_indices
        prediction = (np.stack(array_list) for array_list in zip(
            *[self._predict_one_slave(X, i, offset) for i in estimator_indices]  # check boundary
        ))
        return prediction  # see the output in _predict_one_slave

    def _consecutive_count(self, predicted_labels: List[np.array]):
        n = len(predicted_labels[0])
        prediction_points = len(predicted_labels)
        consecutive_labels = np.ones((prediction_points, n))
        for i in range(1, prediction_points):
            equal = predicted_labels[i - 1] == predicted_labels[i]
            consecutive_labels[i, equal] = consecutive_labels[i - 1, equal] + 1
        return consecutive_labels  # prediction_points x n_instances

    def predict_proba(self, *args):
        """
        Args:
            X (np.array): input features
        Returns:
            predictions as a numpy array of shape (2, n_selected_estimators, n_instances, n_classes)
            where first subarray stands for probas, and second for scores
        """
        predicted_probas, scores, *_ = args 
        if self.transform_score:
            scores = self._transform_score(scores)
        scores = np.tile(scores[..., None], (1, 1, self.n_classes))
        prediction = np.stack([predicted_probas, scores], axis=0)
        if prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)
        return prediction

    def predict(self, X):
        """
        Args:
            X (np.array): input features
        Returns:
            predictions as a numpy array of shape (2, n_selected_estimators, n_instances)
            where first subarray stands for labels, and second for scores
        """
        prediction = self.predict_proba(X)
        labels = prediction[0:1].argmax(-1)
        scores = prediction[1:2, ..., 0]
        prediction = np.stack([labels, scores], 0)
        if prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)
        return prediction

    def _score(self, X, y, accuracy_importance=None, training=True):
        y = np.array(y).flatten()
        accuracy_importance = accuracy_importance or self.accuracy_importance
        predictions = self._predict(X, training)[0]
        prediction_points = predictions.shape[0]
        accuracies = (predictions == np.tile(y, (prediction_points, 1))).sum(axis=1) / len(y)
        return (1 - accuracy_importance) * self.earliness[:prediction_points] + accuracy_importance * accuracies

    def _get_applicable_index(self, last_available_idx):
        idx = np.searchsorted(self.prediction_idx, last_available_idx, side='right')
        if idx == 0:
            raise RuntimeError('Too few points for prediction!')
        idx -= 1
        offset = last_available_idx - self.prediction_idx[idx]
        return idx, offset
