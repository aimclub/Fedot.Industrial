from abc import ABC
from typing import List

import numpy as np
import pandas as pd

from core.operation.utils.analyzer import PerformanceAnalyzer


class AbstractObject(ABC):
    """
    Base class for feature/predictions lists
    """

    def __init__(self):
        pass

    def create(self):
        raise NotImplementedError


class FeatureList(AbstractObject):
    """
    Class responsible for creation of feature list.

    :param list_of_generators: list of generators from config
    :param data: pandas.DataFrame with train/test data
    :param dataset_name: name of dataset
    """

    def __init__(self,
                 list_of_generators: list,
                 data: pd.DataFrame,
                 dataset_name: str) -> None:
        super().__init__()
        self.list_of_generators = list_of_generators
        self.data = data
        self.dataset_name = dataset_name

    def create(self) -> List[pd.DataFrame]:
        return list(map(lambda x: x.extract_features(self.data, self.dataset_name), self.list_of_generators))


class PredictorList(AbstractObject):
    """
    Class responsible for creation of predictors list.

    :param train_labels_set: numpy.ndarray with train labels
    :param feature_list: list of pandas.DataFrame with features
    :param operation: callable function (fitting Fedot model)
    """

    def __init__(self, train_labels_set: np.ndarray,
                 feature_list: list,
                 operation: callable):
        super().__init__()
        self.train_labels_set = train_labels_set
        self.feature_list = feature_list
        self.operation = operation

    def create(self):
        fedot_models_list = list(map(lambda x: self.operation(x, self.train_labels_set), self.feature_list))
        return fedot_models_list


class PredictionsList(AbstractObject):
    """
    Class responsible for creation of predictions list for predictors and features.

    :param predictor_list: list of predictors (Fedot models)
    :param feature_list: list of pandas.DataFrame with features
    :param operation: callable function to obtain predictions
    """

    def __init__(self, predictor_list, feature_list, operation):
        super().__init__()
        self.operation = operation
        self.predictor_list = predictor_list
        self.feature_list = feature_list

    def create(self):
        if self.operation == 'predictions':
            return list(map(lambda x, y: x.predict(y), self.predictor_list, self.feature_list))
        elif self.operation == 'predictions_proba':
            return list(map(lambda x, y: x.predict_proba(y), self.predictor_list, self.feature_list))


class MetricsDict:
    """
    Class responsible for creation metrics dict based on predictions_list,
    predictions probability list and target. Apply methods of
    PerformanceAnalyzer class for chosen metrics:
    ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'].

    :param predictions_list: list of numpy.ndarray with predictions
    :param predictions_proba_list: list of numpy.ndarray with predictions probability
    :param target: numpy.ndarray with target
    """

    def __init__(self, predictions_list, predictions_proba_list, target):
        self.predictions_list = predictions_list
        self.predictions_proba_list = predictions_proba_list
        self.target = target
        self.analyzer = PerformanceAnalyzer()
        self.metrics_name = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']

    def create(self):
        return list(map(lambda x, y: self.analyzer.calculate_metrics(self.metrics_name,
                                                                     target=self.target,
                                                                     predicted_labels=x,
                                                                     predicted_probs=y),
                        self.predictions_list,
                        self.predictions_proba_list))
