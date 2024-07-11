from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.early_tc.base_early_tc import BaseETC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

class EconomyK(BaseETC):
    def __init__(self, params: Optional[OperationParameters] = None):    
        if params is None:
            params = {}    
        super().__init__(params)
        self.prediction_mode = params.get('prediction_mode', 'last_available')
        self.lambda_ = params.get('lambda', 1.)
        self._cluster_factor = params.get('cluster_factor' , 1)
        # self.confidence_mode = params.get('confidence_mode', 'time') # or 'confidence'
        self._random_state = 2104
        self.__cv = 5

    def _init_model(self, X, y):
        super()._init_model(X, y)
        self.n_clusters = int(self._cluster_factor * self.n_classes)
        self._clusterizer = KMeans(self.n_clusters, random_state=self._random_state)
        self.state = np.zeros((self.n_pred, self.n_clusters, self.n_classes, self.n_classes)) 

    def fit(self, X, y):
        y = y.flatten().astype(int)
        self._init_model(X, y)
        self._pyck_ = confusion_matrix(y, self._clusterizer.fit(X).labels_, normalize='true')[:self.n_classes, :self.n_clusters]
        for i in range(self.n_pred):
            self._fit_one_interval(X, y, i)
    
    def _fit_one_interval(self, X, y, i):
        X_part = X[..., :self.prediction_idx[i] + 1]
        X_part = self.scalers[i].fit_transform(X_part)
        y_pred = cross_val_predict(self.slave_estimators[i], X_part, y, cv=self.__cv)
        self.slave_estimators[i].fit(X_part, y)
        states_by_i = np.zeros(( self.n_clusters, self.n_classes, self.n_classes))
        np.add.at(states_by_i, (self._clusterizer.labels_, y, y_pred), 1)
        states_by_i /= np.mean(states_by_i, -2, keepdims=True)
        states_by_i[np.isnan(states_by_i)] = 0
        states_by_i[:, np.eye(self.n_classes).astype(bool)] = 0
        self.state[i] = states_by_i

    def _predict_one_slave(self, X, i, offset=0):
        cluster_centers = self._clusterizer.cluster_centers_[:, :self.prediction_idx[i] + 1] # n_clust x len
        X_part = X[..., max(0, offset - 1):self.prediction_idx[i] + 1]  # n_inst x len
        X_part = self.scalers[i].transform(X_part)
        probas = self.slave_estimators[i].predict_proba(X_part)
        optimal_time, is_optimal = self._get_prediction_time(X_part, cluster_centers, i) 
        return probas, optimal_time, is_optimal
    
    def __cluster_probas(self, X, centroids):
        length = centroids.shape[-1]
        diffs = np.subtract.outer(X, centroids).swapaxes(1, 2)
        diffs = diffs[..., np.eye(length).astype(bool)] # n_inst x n_clust x len
        distances = np.linalg.norm(diffs, axis=-1) 
        delta_k = 1. - distances / distances.mean(axis=-1)[:, None]
        s = 1. / (1. + np.exp(-self.lambda_ * delta_k)) 
        return s / s.sum(axis=-1)[:, None] # n_inst x n_clust

    def __expected_costs(self, X, cluster_centroids, i):
        cluster_probas = self.__cluster_probas(X, cluster_centroids) # n_inst x n_clust
        s_glob = np.sum(np.transpose(
                np.sum(self.state[i:], axis=-1), axes=(0, 2, 1)
            ) * self._pyck_[None, ...], axis=1)
        costs = cluster_probas @ s_glob.T # n_inst x time_left
        costs -= self.earliness[None, i:] * (1 - self.accuracy_importance) # subtract or add ?
        return costs

    def _get_prediction_time(self, X, cluster_centroids, i):
        costs = self.__expected_costs(X, cluster_centroids, i) 
        min_costs = np.argmin(costs, axis=-1)
        is_optimal = min_costs == 0
        time_optimal = self.prediction_idx[min_costs + i]
        return time_optimal, is_optimal # n_inst
    
    def predict_proba(self, X):
        probas, times, _ = self._predict(X, training=False)
        return super().predict_proba(probas, times)

    def _transform_score(self, time):
        idx = self._estimator_for_predict[-1]
        scores = (1 - (time - self.prediction_idx[idx]) / self.prediction_idx[-1])  # [1 / n; 1 ] - 1 / n) * n /(n - 1) * 2 - 1
        n = self.n_pred
        scores -= 1 / n
        scores *= n / (n - 1) * 2
        scores -= 1
        return scores


