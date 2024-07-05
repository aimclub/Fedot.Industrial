from typing import Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

class ECEC(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        
    def _init_model(self, X, y):
        super()._init_model(X, y)
        self._reliabilities = np.zeros((self.n_pred, self.n_classes, self.n_classes))

    def _predict_one_slave(self, X, i, offset=0):
        predicted_probas, predicted_labels = super()._predict_one_slave(X, i, offset) 
        reliabilities = self._reliabilities[i, predicted_labels, predicted_labels].flatten() # n_inst
        return predicted_labels.astype(int), predicted_probas, reliabilities
    
    def _predict(self, X, training=False):
        predicted_labels, predicted_probas, reliabilities = super()._predict(X, training)
        reliabilities = np.stack(reliabilities)
        confidences = 1 - np.cumprod(1 - reliabilities, axis=0)
        non_confident = confidences < self.confidence_thresholds[:len(predicted_labels), None]
        return predicted_labels, predicted_probas, non_confident, confidences
    
    def predict(self, X):
        predicted_labels, _, non_confident, confidences = self._predict(X)
        predicted_labels = np.stack(predicted_labels)
        predicted_labels[non_confident] = -1
        return predicted_labels, confidences
    
    def predict_proba(self, X):
        _, predicted_probas, non_confident, confidences = self._predict(X)
        predicted_probas = np.stack(predicted_probas)
        predicted_probas[non_confident] = -1
        return predicted_probas, confidences

    def _score(self, X, y, alpha, training=False):
        y = y.astype(int)
        predicted_labels, *_ = super()._predict(X, training) # n_pred x n_inst 
        predicted_labels = np.stack(predicted_labels)
        n = predicted_labels.shape[0]
        accuracies = (predicted_labels == np.tile(y, (n, 1))) # n_pred x n_inst
        confidences = np.ones((n, X.shape[0]), dtype='float32')
        for i in range(n):
            y_pred = predicted_labels[i]
            reliability_i = confusion_matrix(y, y_pred, normalize='pred')
            confidences[i] = 1 - reliability_i[y, y_pred] # n_inst
            self._reliabilities[i] = reliability_i
        confidences = 1 - np.cumprod(confidences, axis=0) # n_pred x n_inst
        candidates = self._select_thrs(confidences) # n_candidates
        cfs = np.zeros((len(candidates), n))
        for i, candidate in enumerate(candidates):
            mask = confidences >= candidate  # n_pred x n_inst
            accuracy_for_candidate = (accuracies * mask).sum(1) / mask.sum(1) # n_pred
            cfs[i] = self.cost_func(self.earliness, accuracy_for_candidate, alpha)
        self._best_estimator_idx = np.argmin(cfs.mean(0))
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
        super().fit(X, y)
        self.confidence_thresholds = self._score(X, y, self.accuracy_importance, training=True)
    