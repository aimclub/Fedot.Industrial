from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.early_tc.base_early_tc import EarlyTSClassifier

class ProbabilityThresholdClassifier(EarlyTSClassifier):
    f"""
    Two-tier Early time-series classification model 
    uniting consecutive prediction comparison and thresholding by predicted probability.
    """
    def __init__(self, params: Optional[OperationParameters] = {}): 
        super().__init__(params)
        self.probability_threshold = params.get('probability_threshold', None)

    def _init_model(self, X, y):
        super()._init_model(X, y)
        if self.probability_threshold is None:
            self.probability_threshold = 1 / len(self.classes_[0])
        eps = 1e-7
        if self.probability_threshold == 1:
            self.probability_threshold -= eps
        if self.probability_threshold == 0:
            self.probability_threshold += eps
    
    def predict_proba(self, X):
        _, predicted_probas, non_acceptance = self._predict(X, training=False)
        scores = predicted_probas.max(-1)
        scores[~non_acceptance & (scores < self.probability_threshold)] = self.probability_threshold + \
            (1 - self.probability_threshold) * self.consecutive_predictions / self.n_pred
        predicted_probas[non_acceptance] = 0
        return super().predict_proba(predicted_probas, scores)

    def _predict(self, X, training=True):
        predicted_probas, predicted_labels = super()._predict(X, training)
        non_acceptance = self._consecutive_count(predicted_labels) < self.consecutive_predictions
        double_check = predicted_probas.max(axis=-1) > self.probability_threshold
        non_acceptance[non_acceptance & double_check] = False
        return predicted_labels, predicted_probas, non_acceptance

    def _score(self, X, y, accuracy_importance=None):
        scores = super()._score(X, y, accuracy_importance)
        self._chosen_estimator_idx = np.argmax(scores)
        return scores
    
    def fit(self, X, y):
        super().fit(X, y)
        self._score(X, y, self.accuracy_importance)
    
    def _transform_score(self, confidences):
        thr = self.probability_threshold
        confidences = confidences - thr
        positive = confidences > 0
        confidences[positive] *= 1 / (1 - thr)
        confidences[~positive] *= 1 / thr
        return confidences
