from typing import Union, List, Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.abstraction.decorators import convert_to_input_data
from fedot.core.data.data import InputData, OutputData
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin, BaseEstimator
from sktime.classification.dictionary_based import MUSE, WEASEL
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC

class ProbabilityThresholdClassifier(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):
        if params is None:
            params = {}    
        super().__init__()
        self.probability_threshold = params.get('probability_threshold', 0.85)
    
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
        non_acceptance = self._consecutive_count(predicted_labels) < self.acceptance_threshold
        to_second_check = np.argwhere(non_acceptance)
        predicted_probas = np.stack(predicted_probas)
        predicted_labels = np.stack(predicted_labels)
        # for each point of estimation 
        for i in range(predicted_labels.shape[0]):
            # find not accepted points
            ith_point_to_oc = to_second_check[to_second_check[:, 0] == i, 1]
            # if they are not outliers
            final_verdict = (predicted_probas[i, ith_point_to_oc] > self.acceptance_threshold).any()
            # mark as accepted
            non_acceptance[i, np.argwhere(final_verdict >= 0).flatten()] = False
        return predicted_labels, predicted_probas, non_acceptance
