from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.early_tc.base_early_tc import EarlyTSClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM


class TEASER(EarlyTSClassifier):
    """
     Two-tier Early and Accurate Series classifiER

     from “TEASER: early and accurate time series classification,”
           Data Min. Knowl. Discov., vol. 34, no. 5, pp. 1336–1362, 2020
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self._oc_svm_params = (100., 10., 5., 2.5, 1.5, 1., 0.5, 0.25, 0.1)

    def _init_model(self, X, y):
        super()._init_model(X, y)
        self.oc_estimators = [None] * self.n_pred

    def _fit_one_interval(self, X, y, i):
        probas = super()._fit_one_interval(X, y, i)
        filtered_probas = self._filter_trues(probas, y)
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

    def _filter_trues(self, predicted_probas, y):  # different logic in sktime
        predicted_labels = np.argmax(predicted_probas, axis=-1).flatten()
        return predicted_probas[predicted_labels == y]

    def _form_X_oc(self, predicted_probas):
        d = (predicted_probas.max() - predicted_probas)
        d[d == 0] = 1
        d = d.min(axis=-1).reshape(-1, 1)
        return np.hstack([predicted_probas, d])

    def _predict(self, X, training=False):
        estimator_indices, offset = self._select_estimators(X)
        X_ocs, predicted_probas, predicted_labels = map(np.stack, zip(
            *[self._predict_one_slave(X, i, offset) for i in estimator_indices]
        ))
        non_acceptance = self._consecutive_count(predicted_labels) < self.consecutive_predictions
        final_verdicts = np.zeros((len(estimator_indices), X.shape[0]))
        # for each point of estimation
        for i in range(predicted_labels.shape[0]):
            # find not accepted points
            X_to_ith = X_ocs[i]
            # if they are not outliers
            final_verdict = self.oc_estimators[estimator_indices[i]].decision_function(X_to_ith)
            # mark as accepted
            final_verdicts[i] = final_verdict
        (non_acceptance[non_acceptance & (final_verdict > 0)],
         final_verdicts[non_acceptance],
         final_verdicts[~non_acceptance & (final_verdicts < 0)]
         ) = False, -1, self.consecutive_predictions / self.n_pred
        return predicted_labels, predicted_probas, non_acceptance, final_verdicts

    def predict_proba(self, X):
        _, predicted_probas, non_acceptance, final_verdicts = self._predict(X)
        predicted_probas[non_acceptance] = 0  # final_verdicts[non_acceptance, None]
        return super().predict_proba(predicted_probas, final_verdicts)

    def _score(self, X, y, accuracy_importance=None):
        scores = super()._score(X, y, accuracy_importance)
        self._chosen_estimator_idx = np.argmax(scores)
        return scores

    def fit(self, X, y):
        super().fit(X, y)
        return self._score(X, y, self.accuracy_importance)

    def _transform_score(self, scores):
        return np.tanh(scores)
