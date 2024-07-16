from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


class ECEC(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.__cv = 5
        
    def _init_model(self, X, y):
        super()._init_model(X, y)
        self._reliabilities = np.zeros((self.n_pred, self.n_classes, self.n_classes))

    def _predict_one_slave(self, X, i, offset=0):
        predicted_probas, predicted_labels = super()._predict_one_slave(X, i, offset) 
        reliabilities = self._reliabilities[i, predicted_labels, predicted_labels].flatten() # n_inst
        return predicted_labels.astype(int), predicted_probas, reliabilities
    
    def _predict(self, X, training=False):
        predicted_labels, predicted_probas, reliabilities = super()._predict(X, training)
        confidences = 1 - np.cumprod(1 - reliabilities, axis=0)
        non_confident = confidences < self.confidence_thresholds[:len(predicted_labels), None]
        return predicted_labels, predicted_probas, non_confident, confidences
    
    def predict_proba(self, X):
        _, predicted_probas, non_confident, confidences = self._predict(X)
        predicted_probas[non_confident] = -1
        return super().predict_proba(predicted_probas, confidences)
    
    def _fit_one_interval(self, X, y, i):
        X_part = X[..., :self.prediction_idx[i] + 1]
        X_part = self.scalers[i].fit_transform(X_part)
        self.slave_estimators[i].fit(X_part, y)
        labels = cross_val_predict(self.slave_estimators[i], X_part, y, cv=self.__cv)
        return labels

    def _score(self, y, y_pred, alpha):
        matches = (y_pred == np.tile(y, (self.n_pred, 1))) # n_pred x n_inst
        n, n_inst, *_ = matches.shape
        confidences = np.ones((n, n_inst), dtype='float32')
        for i in range(self.n_pred):
            confidences[i] = self._reliabilities[i, y, y_pred[i]]
        confidences = 1 - np.cumprod(1 - confidences, axis=0) # n_pred x n_inst
        candidates = self._select_thrs(confidences) # n_candidates
        cfs = np.zeros((len(candidates), n))
        for i, candidate in enumerate(candidates):
            mask = confidences >= candidate  # n_pred x n_inst
            accuracy_for_candidate = (matches * mask).sum(1) / mask.sum(1) # n_pred
            cfs[i] = self.cost_func(self.earliness, accuracy_for_candidate, alpha)
        self._chosen_estimator_idx = np.argmin(cfs.mean(0))
        return candidates[np.argmin(cfs, axis=0)] # n_pred

    @staticmethod
    def _select_thrs(confidences):
        C = np.unique(confidences.round(3))
        difference = np.diff(C)
        pair_means = C[:-1] + difference / 2
        difference_shifted = np.roll(difference, 1)
        difference_idx = np.argwhere(difference <= difference_shifted)
        means_candidates = pair_means[difference_idx].flatten()    
        return means_candidates if len(means_candidates) else C
        
    @staticmethod
    def cost_func(earliness, accuracies, alpha):
        return alpha * (1 - accuracies) + (1 - alpha) * earliness
    
    def fit(self, X, y):
        y = np.array(y).flatten().astype(int)
        self._init_model(X, y)
        labels = []
        for i in range(self.n_pred):
            labels.append(self._fit_one_interval(X, y, i))
        predicted_labels = np.stack(labels)
        for i in range(self.n_pred):
            y_pred = predicted_labels[i]
            reliability_i = confusion_matrix(y, y_pred, normalize='pred')
            self._reliabilities[i] = reliability_i
        self.confidence_thresholds = self._score(y, predicted_labels, self.accuracy_importance)

    def _transform_score(self, confidences):
        thr = self.confidence_thresholds[self._estimator_for_predict[-1]]
        confidences = confidences - thr
        positive = confidences > 0
        confidences[positive] *= 1 / (1 - thr)
        confidences[~positive] *= 1 / thr
        return confidences
    