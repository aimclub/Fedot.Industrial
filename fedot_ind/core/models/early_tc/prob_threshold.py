from typing import Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC

class ProbabilityThresholdClassifier(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):
        if params is None:
            params = {}    
        super().__init__()
        self.probability_threshold = params.get('probability_threshold', None)

    def _init_model(self, X, y):
        super()._init_model(X, y)
        if self.probability_threshold is None:
            self.probability_threshold = 1 / len(self.classes_[0])
    
    def predict_proba(self, X):
        _, predicted_probas, non_acceptance = self._predict(X)
        predicted_probas[non_acceptance] = 0
        return predicted_probas.squeeze()
        
    def predict(self, X):
        predicted_labels, _, non_acceptance = self._predict(X)
        predicted_labels[non_acceptance] = -1
        # predicted_labels[non_acceptance] = final_verdicts[non_acceptance]
        return predicted_labels # prediction_points x n_instances

    def _predict(self, X):
        predicted_labels, predicted_probas = super()._predict(X)
        predicted_probas = np.stack(predicted_probas)
        predicted_labels = np.stack(predicted_labels)
        non_acceptance = self._consecutive_count(predicted_labels) < self.consecutive_predictions
        double_check = predicted_probas.max(axis=-1) > self.probability_threshold
        non_acceptance[non_acceptance & double_check] = False
        return predicted_labels, predicted_probas, non_acceptance