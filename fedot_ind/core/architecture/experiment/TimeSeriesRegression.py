import logging
from typing import Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.decomposition import PCA

from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer


class TimeSeriesRegression:

    def __init__(self, params: Optional[OperationParameters] = None):
        self.strategy = params.get('strategy', 'statistical')
        self.model_hyperparams = params.get('model_params')
        self.generator_runner = params.get('generator_class')
        self.dataset_name = params.get('dataset')
        self.output_folder = params.get('output_folder', None)
        self.use_cache = params.get('use_cache', False)

        self.logger = logging.getLogger('TimeSeriesClassifier')
        self.pca = PCA(n_components=params.get('explained_variance', 0.9), svd_solver='full')

        self.logger.info('TimeSeriesRegression solver initialised')

    def __check_multivariate_data(self, data: pd.DataFrame) -> bool:
        """Method for checking if the data is multivariate.

        Args:
            X: pandas DataFrame with features

        Returns:
            True if data is multivariate, False otherwise

        """
        if isinstance(data.iloc[0, 0], pd.Series):
            return True
        else:
            return False

    def _init_input_data(self, X: pd.DataFrame, target: np.ndarray) -> InputData:
        """Method for initialization of InputData object from pandas DataFrame and numpy array with target values.

        Args:
            X: pandas DataFrame with features
            y: numpy array with target values

        Returns:
            InputData object convenient for FEDOT framework

        """
        y = np.array([float(i) for i in target])

        is_multivariate_data = self.__check_multivariate_data(X)
        if is_multivariate_data:
            input_data = InputData(idx=np.arange(len(X)),
                                   features=np.array(X.values.tolist()),
                                   target=y.reshape(-1, 1),
                                   task=Task(TaskTypesEnum.regression),
                                   data_type=DataTypesEnum.image)
        else:
            input_data = InputData(idx=np.arange(len(X)),
                                   features=X.values,
                                   target=np.ravel(y).reshape(-1, 1),
                                   task=Task(TaskTypesEnum.regression),
                                   data_type=DataTypesEnum.table)

        return input_data

    def fit_regressor(self, features, target):
        self.logger.info('Start fitting regressor')
        self.predictor = Fedot(**self.model_hyperparams)
        self.predictor.fit(features, target)
        return self.predictor

    def fit(self, features,
            target: np.ndarray = None,
            **kwargs) -> object:
        """
        Method for fitting pipeline on train data. It also tunes pipeline and updates it with categorical features.

        Args:
            features: pandas DataFrame with features
            target: numpy array with target values

        Returns:
            fitted FEDOT model as object of ``Pipeline`` class

        """

        self.logger.info('Start fitting pipeline')
        self.train_target = np.array([float(i) for i in target])
        self.train_input_data = self._init_input_data(features, target)
        extracted_train_features = self.generator_runner.transform(input_data=self.train_input_data,
                                                         use_cache=self.use_cache)
        train_size = extracted_train_features.features.shape
        self.train_features = extracted_train_features.features.reshape(train_size[0], train_size[1] * train_size[2])

        self.logger.info('Start applying PCA')
        self.pca_train_features = self.pca.fit_transform(self.train_features)
        self.logger.info(f'PCA n_components left: {self.pca.n_components_}')

        self.fit_regressor(self.pca_train_features, self.train_target)

        return self.predictor

    def predict(self, features: pd.DataFrame, target: np.array):
        """Method for prediction on test data.

        Args:
            features: pandas DataFrame with features
            target: numpy array with target values

        Returns:
            dict with predicted values and probabilities

        """
        self.logger.info('Start prediction')
        self.test_target = np.array([float(i) for i in target])
        self.test_input_data = self._init_input_data(features, target)
        extracted_test_features = self.generator_runner.transform(input_data=self.test_input_data,
                                                             use_cache=self.use_cache)
        test_size = extracted_test_features.features.shape
        self.test_features = extracted_test_features.features.reshape(test_size[0], test_size[1] * test_size[2])

        self.pca_test_features = self.pca.transform(self.test_features)
        self.predicted_labels = self.predictor.predict(features=self.pca_test_features)

        return self.predicted_labels

    def get_metrics(self, target, metric_names: Optional[list] = None):
        analyzer = PerformanceAnalyzer()
        return analyzer.calculate_metrics(target=self.test_target,
                                          predicted_labels=self.predicted_labels,
                                          predicted_probs=None,
                                          target_metrics=metric_names)
