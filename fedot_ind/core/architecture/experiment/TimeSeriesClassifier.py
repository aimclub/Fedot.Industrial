import logging
from typing import List, Union
from typing import Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.api.utils.input_data import init_input_data
from fedot_ind.api.utils.saver_collections import ResultSaver
from fedot_ind.core.metrics.evaluation import PerformanceAnalyzer


class TimeSeriesClassifier:
    """Class responsible for interaction with Fedot classifier.

    Args:
        params: which are ``generator_name``, ``generator_runner``, ``model_hyperparams``,
                ``ecm_model_flag``, ``dataset_name``, ``output_dir``.


    Attributes:
        logger (Logger): logger instance
        predictor (Fedot): Fedot model instance
        y_train (np.ndarray): target for training
        train_features (pd.DataFrame): features for training

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.strategy = params.get('strategy')
        self.model_hyperparams = params.get('model_params')
        self.generator_runner = params.get('generator_class')
        self.dataset_name = params.get('dataset')
        self.output_folder = params.get('output_folder', None)

        self.saver = ResultSaver(dataset_name=self.dataset_name,
                                 generator_name=self.strategy,
                                 output_dir=self.output_folder)
        self.logger = logging.getLogger('TimeSeriesClassifier')
        self.datacheck = DataCheck()

        self.prediction_proba = None
        self.test_predict_hash = None
        self.prediction_label = None
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None

        self.logger.info('TimeSeriesClassifier initialised')

    def fit(self, features: Union[np.ndarray, pd.DataFrame],
            target: np.ndarray,
            **kwargs) -> object:

        baseline_type = kwargs.get('baseline_type', None)
        self.logger.info(f'Fitting model')
        self.y_train = target
        if self.generator_runner is None:
            raise AttributeError('Feature generator is not defined')

        input_data = init_input_data(features, target)
        output_data = self.generator_runner.transform(input_data)
        train_features = pd.DataFrame(output_data.predict, columns=self.generator_runner.relevant_features)

        self.train_features = self.datacheck.check_data(input_data=train_features, return_df=True)

        if baseline_type is not None:
            self.predictor = self._fit_baseline_model(self.train_features, target, baseline_type)
        else:
            self.predictor = self._fit_model(self.train_features, target)

        return self.predictor

    def _fit_model(self, features: pd.DataFrame, target: np.ndarray) -> Fedot:
        """Fit Fedot model with feature and target.

        Args:
            features: features for training
            target: target for training

        Returns:
            Fitted Fedot model

        """
        self.predictor = Fedot(**self.model_hyperparams)
        self.predictor.fit(features, target)
        self.logger.info(
            f'Solver fitted: {self.strategy}_extractor -> fedot_pipeline ({self.predictor.current_pipeline})')
        return self.predictor

    def _fit_baseline_model(self, features: pd.DataFrame, target: np.ndarray, baseline_type: str = 'rf') -> Pipeline:
        """Returns pipeline with the following structure:
            ``[initial data] -> [scaling] -> [baseline type]``

        By default, baseline type is random forest, but it can be changed to any other model from Fedot library:
        logit, knn, svc, qda, xgboost.

        Args:
            features: features for training
            target: target for training
            baseline_type: type of baseline model

        Returns:
            Fitted Fedot pipeline with baseline model

        """
        self.logger.info(f'Baseline model pipeline: scaling -> {baseline_type}. Fitting...')
        node_scaling = PrimaryNode('scaling')
        node_final = SecondaryNode(baseline_type,
                                   nodes_from=[node_scaling])
        baseline_pipeline = Pipeline(node_final)
        input_data = init_input_data(features, target)
        baseline_pipeline.fit(input_data)
        self.logger.info(f'Baseline model has been fitted')
        return baseline_pipeline

    def predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        self.prediction_label = self.__predict_abstraction(test_features=features, mode='labels', **kwargs)
        return self.prediction_label

    def predict_proba(self, features: np.ndarray, **kwargs) -> np.ndarray:
        self.prediction_proba = self.__predict_abstraction(test_features=features, mode='probs', **kwargs)
        return self.prediction_proba

    def __predict_abstraction(self,
                              test_features: Union[np.ndarray, pd.DataFrame],
                              mode: str = 'labels', **kwargs):
        self.logger.info(f'Predicting with {self.strategy} generator')

        if self.test_features is None:
            input_data = init_input_data(test_features, kwargs.get('target'))
            output_data = self.generator_runner.transform(input_data)
            test_features = pd.DataFrame(output_data.predict, columns=self.generator_runner.relevant_features)
            self.test_features = self.datacheck.check_data(input_data=test_features, return_df=True)

        if isinstance(self.predictor, Pipeline):
            self.input_test_data = init_input_data(self.test_features, kwargs.get('target'))
            prediction_label = self.predictor.predict(self.input_test_data, output_mode=mode).predict
            return prediction_label
        else:
            if mode == 'labels':
                prediction_label = self.predictor.predict(self.test_features)
            else:
                prediction_label = self.predictor.predict_proba(self.test_features)
            return prediction_label

    def get_metrics(self, target: Union[np.ndarray, pd.Series], metric_names: Union[str, List[str]]):
        analyzer = PerformanceAnalyzer()
        return analyzer.calculate_metrics(target=target,
                                          predicted_labels=self.prediction_label,
                                          predicted_probs=self.prediction_proba,
                                          target_metrics=metric_names)

    def save_prediction(self, predicted_data: np.ndarray, kind: str):
        self.saver.save(predicted_data, kind)

    def save_metrics(self, metrics: dict):
        self.saver.save(metrics, 'metrics')
