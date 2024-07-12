import numpy as np
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from fedot_ind.api.utils.data import init_input_data


class SklearnCompatibleClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for FedotIndustrial to make it compatible with sklearn.

    Args:
        estimator (Pipeline): FedotIndustrial pipeline.

    """

    def __init__(self, estimator: Pipeline):
        self.estimator = estimator
        self.classes_ = None

    def fit(self, X, y):
        self.estimator.fit(init_input_data(X, y))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        labels = self.estimator.predict(init_input_data(X, None)).predict
        return labels

    def predict_proba(self, X):
        probs = self.estimator.predict(init_input_data(X, None), output_mode='probs').predict
        return probs
