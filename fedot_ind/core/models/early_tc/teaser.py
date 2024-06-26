from typing import Union, List, Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData, OutputData
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sktime.classification.dictionary_based import MUSE, WEASEL
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class TEASER(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):        
        super().__init__()
        if params is None:
            params = {}
        self.prediction_mode = params.get('prediction_mode', 'best_by_harmonic_mean')
        self.interval_length = params.get('interval_length', 10) # rewrite as interval_length
        self.acceptance_threshold = params.get('acceptance_threshold', 5)
        self.hm_shift_to_acc = params.get('hm_shift_to_acc', 1.)
        assert self.acceptance_threshold < self.interval_length, 'Not enough checkpoints for prediction proof'

        # how to pass into ? % what needed
        self._oc_svm_params = [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]
        self.weasel_params = {}
        self.random_state = None # is needed?

    def _init_model(self, max_data_length):
        self.prediction_idx = self._compute_prediction_points(max_data_length)
        self.n_pred = len(self.prediction_idx)
        self.oc_estimators = [None] * self.n_pred
        self.slave_estimators = [WEASEL(random_state=self.random_state, support_probabilities=True, **self.weasel_params) for _ in range(self.n_pred)]
        self.scalers = [StandardScaler() for _ in range(self.n_pred)]
        self.__offset = max_data_length % self.interval_length
        self.best_estimator_idx = -1

    def fit(self, input_data: InputData):
        input_data = self.__convert_pd(input_data)
        X, y = input_data.features, input_data.target # what's passed in case of classification to training? 
        self._init_model(max_data_length=X.shape[-1])
        for i in range(self.n_pred):
            self._fit_one_interval(X, y, i)
        self.best_estimator_idx = np.argmax(self._score(X, y, self.hm_shift_to_acc))

    def _fit_one_interval(self, X, y, i):
        X_part = X[..., :self.prediction_idx[i] + 1] # what's dimensionality of input? will it work in case of multivariate?
        X_part = self.scalers[i].fit_transform(X_part)
        probas = self.slave_estimators[i].fit_predict_proba(X_part, y)
        filtered_probas = self._filter_trues(probas, y) #
        X_oc = self._form_X_oc(filtered_probas)
        self.oc_estimators[i] = GridSearchCV(OneClassSVM(),
                                            param_grid={"gamma": self._oc_svm_params},
                                            scoring='accuracy',
                                            cv=min(X.shape[0], 10)
                                ).fit(X_oc, np.ones((len(X_oc), 1))).best_estimator_

    def _predict_one_slave(self, X, i, offset=0):
        X_part = X[..., max(0, offset - 1):self.prediction_idx[i] + 1] 
        X_part = self.scalers[i].transform(X_part)
        probas = self.slave_estimators[i].predict_proba(X_part)
        X_oc = self._form_X_oc(probas)
        return X_oc, np.argmax(probas, axis=-1) 
    
    def _compute_prediction_points(self, n_idx):
        """Computes indices for prediction, includes last index, first interval may be greater"""
        prediction_idx = np.arange(n_idx - 1, -1, -self.interval_length)[::-1]
        self.earliness = 1 - prediction_idx / n_idx
        return prediction_idx
    
    def _filter_trues(self, predicted_probas, y): # different logic in sktime
        predicted_labels = np.argmax(predicted_probas, axis=-1).flatten()
        return predicted_probas[predicted_labels == y]
    
    def _form_X_oc(self, predicted_probas):
        d = (predicted_probas.max() - predicted_probas)
        d[d == 0] = 1
        d = d.min(axis=-1).reshape(-1, 1)
        return np.hstack([predicted_probas, d])
    
    def _predict(self, X):
        n = X.shape[0]
        self.states = np.ones((n, self.n_pred, 2)) # num_consec, class
        if self.prediction_mode == 'best_by_harmonic_mean':
            estimator_indices = [self.best_estimator_idx]
        else:
            last_idx, offset = self._get_applicable_index(X.shape[-1] - 1)
            estimator_indices = list(range(last_idx + 1))
        X_ocs, predicted_labels = zip(
            *[self._predict_one_slave(X, i, offset) for i in estimator_indices] # check boundary
        )
        non_acceptance = self._consecutive_count(predicted_labels) < self.acceptance_threshold
        to_oc_check = np.argwhere(non_acceptance)
        X_ocs = np.stack(X_ocs)
        predicted_labels = np.stack(predicted_labels)
        # for each point of estimation 
        for i in range(predicted_labels.shape[0]):
            # find not accepted points
            ith_point_to_oc = to_oc_check[to_oc_check[:, 0] == i, 1]
            X_to_ith = X_ocs[i][ith_point_to_oc]
            # if they are not outliers
            final_verdict = self.oc_estimators[i].predict(X_to_ith) # 1 for accept -1 for reject
            # mark as accepted
            non_acceptance[i, np.argwhere(final_verdict == 1).flatten()] = False
        predicted_labels[non_acceptance] = -1
        return predicted_labels # prediction_points x n_instances

    def _consecutive_count(self, predicted_labels: List[np.array]):
        n = len(predicted_labels[0])
        prediction_points = len(predicted_labels)
        consecutive_labels = np.ones((prediction_points, n))
        for i in range(1, prediction_points):
            equal = predicted_labels[i - 1] == predicted_labels[i]
            consecutive_labels[i, equal] = consecutive_labels[i - 1, equal] + 1
        return consecutive_labels # prediction_points x n_instances 
    
    def __convert_pd(self, input_data):
        if hasattr(input_data.features, 'values'):
            input_data.features = input_data.features.values
        if hasattr(input_data.target, 'values'):
            input_data.target = input_data.target.values
        return input_data
    
    def predict(self, input_data: InputData) -> OutputData:
        input_data = self.__convert_pd(input_data)
        prediction = self._predict(input_data.features)
        return self._convert_to_output(input_data, predict=prediction)

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        return self.predict(input_data)  

    def _score(self, X, y, hm_shift_to_acc=None):
        hm_shift_to_acc = hm_shift_to_acc or self.hm_shift_to_acc
        predictions = self._predict(X)
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
