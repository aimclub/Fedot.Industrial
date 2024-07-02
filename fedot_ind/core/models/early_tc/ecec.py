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
        self._confidences = np.ones((X.shape[0], self.n_pred))
    
    def _score(self, X, y, alpha):
        y = y.astype(int)
        predicted_labels = np.stack(super()._predict(X)[0]).astype(int) # n_pred x n_inst
        n = predicted_labels.shape[0]
        accuracies = (predicted_labels == np.tile(y, (1, n))) # n_pred x n_inst
        confidences = np.ones((n, X.shape[0]), dtype='float32')
        for i in range(n):
            y_pred = predicted_labels[i]
            reliability_i = confusion_matrix(y, y_pred, normalize='pred')
            confidences[i] = 1 - reliability_i[y, y_pred] # n_inst
        confidences = 1 - np.cumprod(confidences, axis=0) # n_pred x n_inst
        candidates = self._select_thrs(confidences) # n_candidates
        cfs = np.zeros_like(candidates)
        for i, candidate in enumerate(candidates):
            mask = confidences >= candidate  # n_pred x n_inst
            accuracy_for_candidate = (accuracies * mask).sum(1) / mask.sum(1) # n_pred
            cfs[i] = self.cost_func(self.earliness, accuracy_for_candidate, alpha)
        return candidates[np.argmin(cfs)]

    @staticmethod
    def _select_thrs(confidences):
        C = np.unique(confidences.round(3))
        difference = np.diff(C)
        pair_means = C[:-1] + difference / 2
        difference_shifted = np.roll(difference, 1)
        difference_idx = np.argwhere(difference > difference_shifted)
        return pair_means[difference_idx].flatten()        
        
    @staticmethod
    def cost_func(earliness, accuracies, alpha):
        return alpha * accuracies + (1 - alpha) * earliness
    
    def fit(self, X, y):
        self.confidence_threshold = super().fit(X, y)
    



    





        

