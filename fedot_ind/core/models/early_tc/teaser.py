from typing import Optional
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC


class TEASER(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):        
        super().__init__(params)
        self._oc_svm_params = (100., 10., 5., 2.5, 1.5, 1., 0.5, 0.25, 0.1)

    def _init_model(self, X, y):
        super()._init_model(X, y)
        self.oc_estimators = [None] * self.n_pred

    def _fit_one_interval(self, X, y, i):
        probas = super()._fit_one_interval(X, y, i)
        filtered_probas = self._filter_trues(probas, y) #
        X_oc = self._form_X_oc(filtered_probas)
        self.oc_estimators[i] = GridSearchCV(OneClassSVM(),
                                            param_grid={"gamma": self._oc_svm_params},
                                            scoring='accuracy',
                                            cv=min(X.shape[0], 10)
                                ).fit(X_oc, np.ones((len(X_oc), 1))).best_estimator_

    def _predict_one_slave(self, X, i, offset=0):
        probas, labels = super()._predict_one_slave(X, i, offset)
        X_oc = self._form_X_oc(probas)
        return X_oc, probas, labels
    
    def _filter_trues(self, predicted_probas, y): # different logic in sktime
        predicted_labels = np.argmax(predicted_probas, axis=-1).flatten()
        return predicted_probas[predicted_labels == y]
    
    def _form_X_oc(self, predicted_probas):
        d = (predicted_probas.max() - predicted_probas)
        d[d == 0] = 1
        d = d.min(axis=-1).reshape(-1, 1)
        return np.hstack([predicted_probas, d])
    
    def _predict(self, X):
        estimator_indices, offset = self._select_estimators(X)
        X_ocs, predicted_probas, predicted_labels = zip(
            *[self._predict_one_slave(X, i, offset) for i in estimator_indices] # check boundary
        )
        non_acceptance = self._consecutive_count(predicted_labels) < self.consecutive_predictions
        to_oc_check = np.argwhere(non_acceptance)
        X_ocs = np.stack(X_ocs)
        predicted_probas = np.stack(predicted_probas)
        predicted_labels = np.stack(predicted_labels)
        final_verdicts = np.zeros((len(estimator_indices), X.shape[0]))
        # for each point of estimation 
        for i in range(predicted_labels.shape[0]):
            # find not accepted points
            ith_point_to_oc = to_oc_check[to_oc_check[:, 0] == i, 1]
            X_to_ith = X_ocs[i][ith_point_to_oc]
            # if they are not outliers
            final_verdict = self.oc_estimators[estimator_indices[i]].decision_function(X_to_ith) # 1 for accept -1 for reject
            # mark as accepted
            non_acceptance[i, np.argwhere(final_verdict >= 0).flatten()] = False
            final_verdicts[i] = final_verdict 
        return predicted_labels, predicted_probas, non_acceptance, final_verdicts
    
    def predict_proba(self, X):
        _, predicted_probas, non_acceptance, final_verdicts = self._predict(X)
        predicted_probas[non_acceptance] = final_verdicts[non_acceptance, None]
        return predicted_probas.squeeze()
        
    def predict(self, X):
        predicted_labels, _, non_acceptance, final_verdicts = self._predict(X)
        predicted_labels[non_acceptance] = -1
        # predicted_labels[non_acceptance] = final_verdicts[non_acceptance]
        return predicted_labels # prediction_points x n_instances
    
    def _score(self, X, y, hm_shift_to_acc=None):
        scores = super()._score(X, y, hm_shift_to_acc)
        self._best_estimator_idx = np.argmax(scores)
        return scores
    
    def fit(self, X, y):
        super().fit(X, y)
        return self._score(X, y, self.hm_shift_to_acc)
