from typing import Union, List, Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot.core.data.data import InputData, OutputData
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sktime.classification.dictionary_based import MUSE, WEASEL
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class TEASER(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):        
        super().__init__()
        if params is None:
            params = {}
        self.interval_length = params.get('interval_length', 10) # rewrite as interval_length
        self.acceptance_threshold = params.get('acceptance_threshold', 5)
        self.hm_shift_to_acc = params.get('hm_shift_to_acc', 1.)
        assert self.acceptance_threshold < self.interval_length, 'Not enough checkpoints for prediction proof'
        # how to pass into ? % what needed
        self.oc_svm_params = {}
        self.weasel_params = {}
        self.random_state = None # is needed?

    def _init_model(self, max_data_length):
        self.prediction_idx = self._compute_prediction_points(max_data_length)
        self.n_pred = len(self.prediction_idx)
        self.oc_estimators = [OneClassSVM(**self.oc_svm_params) for _ in range(self.n_pred)]
        self.slave_estimators = [WEASEL(random_state=self.random_state, support_probabilities=True, **self.weasel_params) for _ in range(self.n_pred)]
        self.scalers = [StandardScaler() for _ in range(self.n_pred)] # do we need them separate? no inverse path expected

    def fit(self, input_data: InputData):
        input_data = self.__convert_pd(input_data)
        X, y = input_data.features, input_data.target # what's passed in case of classification to training? 
        self._init_model(max_data_length=X.shape[-1])
        for i in range(self.n_pred):
            self._fit_one_interval(X, y, i)

    def _fit_one_interval(self, X, y, i):
        X_part = X[..., :self.prediction_idx[i]] # what's dimensionality of input? will it work in case of multivariate?
        X_part = self.scalers[i].fit_transform(X_part)
        probas = self.slave_estimators[i].fit_predict_proba(X_part, y)
        filtered_probas = self._filter_positive(probas, y) #
        X_oc = self._form_X_oc(filtered_probas)
        self.oc_estimators[i].fit(X_oc, y)

    def _predict_one_slave(self, X, i):
        X_part = X[..., :self.prediction_idx[i]] 
        X_part = self.scalers[i].transform(X_part)
        probas = self.slave_estimators[i].predict_proba(X_part)
        X_oc = self._form_X_oc(probas)
        return X_oc, np.argmax(probas, axis=-1) 
    
    def _compute_prediction_points(self, n_idx):
        """Computes indices for prediction, includes last index, first interval may be greater"""
        prediction_idx = np.arange(n_idx - 1, -1, -self.interval_length)[::-1]
        self.earliness = 1 - prediction_idx / n_idx
        return prediction_idx
    
    def _filter_positive(self, predicted_probas, y): # different logic in sktime
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
        X_ocs, predicted_labels = zip(
            *[self._predict_one_slave(X, i) for i in range(self.n_pred)]
        )
        non_acceptance = self._consecutive_count(predicted_labels) < self.acceptance_threshold
        to_oc_check = np.argwhere(non_acceptance)
        X_ocs = np.stack(X_ocs)
        predicted_labels = np.stack(predicted_labels)
        # for each point of estimation 
        for i in range(self.n_pred):
            # find not accepted points
            ith_point_to_oc = to_oc_check[to_oc_check[:, 0] == i, 1]
            X_to_ith = X_ocs[i][ith_point_to_oc]
            # if they are not outliers
            final_verdict = self.oc_estimators[i].predict(X_to_ith) # 1 for accept -1 for reject
            # mark as accepted
            non_acceptance[i, np.argwhere(final_verdict == 1).flatten()] = False
        predicted_labels[non_acceptance] = -1
        return predicted_labels

    def _consecutive_count(self, predicted_labels: List[np.array]):
        n = len(predicted_labels[0])
        consecutive_labels = np.ones((self.n_pred, n))
        for i in range(1, self.n_pred):
            equal = predicted_labels[i - 1] == predicted_labels[i]
            consecutive_labels[i, equal] = consecutive_labels[i - 1, equal] + 1
        return consecutive_labels # n_pred x n_instances 
    
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
        accuracies = (predictions == np.tile(y, (1, self.n_pred))).sum(axis=1) / len(y)
        return (1 + hm_shift_to_acc) * accuracies * self.earliness / (hm_shift_to_acc * accuracies + self.earliness)

    def _tune_oc(self):
        #TODO
        pass
