from typing import Optional, List
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClassifierMixin, BaseEstimator
from sktime.classification.dictionary_based import WEASEL
from fedot.core.operations.operation_parameters import OperationParameters


class BaseETC(ClassifierMixin, BaseEstimator):
    def __init__(self, params: Optional[OperationParameters] = None):    
        if params is None:
            params = {}    
        super().__init__()
        self.prediction_mode = params.get('prediction_mode', 'best_by_harmonic_mean')
        self.interval_percentage = params.get('interval_percentage', 10)
        self.consecutive_predictions = params.get('consecutive_predictions', 3)
        self.hm_shift_to_acc = params.get('hm_shift_to_acc', 1.)
        self.random_state = params.get('random_state', None)
        self.weasel_params = {}
        assert self.consecutive_predictions < self.interval_percentage, 'Not enough checkpoints for prediction proof'

    def _init_model(self, X, y):
        max_data_length = X.shape[-1]
        self.prediction_idx = self._compute_prediction_points(max_data_length)
        self.n_pred = len(self.prediction_idx)
        self.slave_estimators = [WEASEL(random_state=self.random_state, support_probabilities=True, **self.weasel_params) for _ in range(self.n_pred)]
        self.scalers = [StandardScaler() for _ in range(self.n_pred)]
        self._best_estimator_idx = -1
        self.classes_ = [np.unique(y)]

    @property
    def required_length(self):
        if not hasattr(self, '_best_estimator_idx'):
            return None
        return self.prediction_idx[self._best_estimator_idx]
    
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
        X_part = X[..., :self.prediction_idx[i] + 1] # what's dimensionality of input? will it work in case of multivariate?
        X_part = self.scalers[i].fit_transform(X_part)
        probas = self.slave_estimators[i].fit_predict_proba(X_part, y)
        return probas

    def _predict_one_slave(self, X, i, offset=0):
        X_part = X[..., max(0, offset - 1):self.prediction_idx[i] + 1] 
        X_part = self.scalers[i].transform(X_part)
        probas = self.slave_estimators[i].predict_proba(X_part)
        return probas, np.argmax(probas, axis=-1) 
    
    def _compute_prediction_points(self, n_idx):
        interval_length = int(n_idx * self.interval_percentage / 100)
        prediction_idx = np.arange(n_idx - 1, -1, -interval_length)[::-1]
        self.earliness = 1 - prediction_idx / n_idx # /n_idx because else the last hm score is always 0
        return prediction_idx
    
    def _select_estimators(self, X, training=False):
        offset = 0
        if not training and self.prediction_mode == 'best_by_harmonic_mean':
            estimator_indices = [self._best_estimator_idx]
        elif training or self.prediction_mode == 'all':
            last_idx, offset = self._get_applicable_index(X.shape[-1] - 1)
            estimator_indices = np.arange(last_idx + 1)
        else:
            raise ValueError('Unknown prediction mode')
        return estimator_indices, offset
    
    def _predict(self, X, training=True):
        estimator_indices, offset = self._select_estimators(X, training)
        prediction = zip(
            *[self._predict_one_slave(X, i, offset) for i in estimator_indices] # check boundary
        )
        return prediction # see the output in _predict_one_slave

    def _consecutive_count(self, predicted_labels: List[np.array]):
        n = len(predicted_labels[0])
        prediction_points = len(predicted_labels)
        consecutive_labels = np.ones((prediction_points, n))
        for i in range(1, prediction_points):
            equal = predicted_labels[i - 1] == predicted_labels[i]
            consecutive_labels[i, equal] = consecutive_labels[i - 1, equal] + 1
        return consecutive_labels # prediction_points x n_instances 
    
    def predict_proba(self, X):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError

    def _score(self, X, y, hm_shift_to_acc=None):
        y = np.array(y).flatten()
        hm_shift_to_acc = hm_shift_to_acc or self.hm_shift_to_acc
        predictions = self._predict(X)[0]
        prediction_points = predictions.shape[0]
        accuracies = (predictions == np.tile(y, (prediction_points, 1))).sum(axis=1) / len(y)
        return (1 + hm_shift_to_acc) * accuracies * self.earliness[:prediction_points] / (hm_shift_to_acc * accuracies + self.earliness[:prediction_points])


    def _get_applicable_index(self, last_available_idx):
        idx = np.searchsorted(self.prediction_idx, last_available_idx, side='right')
        if idx == 0:
            raise RuntimeError('Too few points for prediction!')
        idx -= 1
        offset = last_available_idx - self.prediction_idx[idx]
        return idx, offset
