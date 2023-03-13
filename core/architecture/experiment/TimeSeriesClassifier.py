from typing import Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from core.log import default_log as logger
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from core.api.utils.checkers_collections import DataCheck
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.preprocessing.FeatureBuilder import FeatureBuilderSelector
from core.models.ExperimentRunner import ExperimentRunner


class TimeSeriesClassifier:
    """Class responsible for interaction with Fedot classifier.

    Args:
        generator_name: name of the generator for feature extraction
        generator_runner: generator runner instance for feature extraction
        model_hyperparams: hyperparameters for Fedot model
        ecm_model_flag: flag for error correction model

    Attributes:
        logger (Logger): logger instance
        predictor (Fedot): Fedot model instance
        y_train (np.ndarray): target for training
        train_features (pd.DataFrame): features for training

    """

    def __init__(self,
                 generator_name: str = None,
                 generator_runner: ExperimentRunner = None,
                 model_hyperparams: dict = None,
                 ecm_model_flag: bool = False,
                 dataset_name: str = None, ):
        self.logger = logger(self.__class__.__name__)
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None
        self.dataset_name = dataset_name

        self.datacheck = DataCheck()
        self.generator_name = generator_name
        self.generator_runner = generator_runner
        self.feature_generator_dict = {self.generator_name: self.generator_runner}
        self.model_hyperparams = model_hyperparams
        self.ecm_model_flag = ecm_model_flag

        if self.generator_runner is not None:
            self._init_builder()

        # self.__init_checks()

    def _init_builder(self) -> None:
        """Initialize builder with all operations combining generator name and transformation method.

        """
        for name, runner in self.feature_generator_dict.items():
            self.feature_generator_dict[name] = FeatureBuilderSelector(name, runner).select_transformation()

    def fit(self, raw_train_features: pd.DataFrame,
            train_target: np.ndarray,
            baseline_type: str = None) -> Fedot:

        # self.__fit_abstraction(raw_train_features, train_target, baseline_type)

        self.logger.info(f'Fitting model...')
        self.y_train = train_target
        if self.generator_runner is not None:
            self.train_features = self.generator_runner.extract_features(ts_frame=raw_train_features,
                                                                         dataset_name=self.dataset_name)
        else:
            self.train_features = raw_train_features
        self.train_features = self.datacheck.check_data(self.train_features)

        if baseline_type is not None:
            self.predictor = self._fit_baseline_model(self.train_features, train_target, baseline_type)
        else:
            self.predictor = self._fit_model(self.train_features, train_target)

        return self.predictor

    def predict(self, test_features: np.ndarray, target: np.ndarray) -> dict:
        self.prediction_label = self.__predict_abstraction(test_features=test_features, mode='labels')

        return self.prediction_label

    def predict_proba(self, test_features: np.ndarray, target: np.ndarray) -> dict:
        self.prediction_proba = self.__predict_abstraction(test_features=test_features, mode='probs')

        return self.prediction_proba

    def get_metrics(self, target: np.ndarray, metric_names) -> dict:
        metrics_dict = PerformanceAnalyzer().calculate_metrics(target=target,
                                                               predicted_labels=self.prediction_label,
                                                               predicted_probs=self.prediction_proba,
                                                               target_metrics=metric_names)

        return metrics_dict

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
        input_data = array_to_input_data(features_array=features,
                                         target_array=target)
        baseline_pipeline.fit(input_data)
        self.logger.info(f'Baseline model has been fitted')
        return baseline_pipeline

    # def __fit_abstraction(self,
    #                       raw_train_features: Union[np.ndarray, pd.DataFrame],
    #                       train_target: np.ndarray,
    #                       baseline_type: str = None):
    #     self.logger.info(f'Fitting model...')
    #     self.y_train = train_target
    #     if self.generator_runner is not None:
    #         self.train_features = self.generator_runner.extract_features(ts_frame=raw_train_features,
    #                                                                      dataset_name=self.dataset_name)
    #     else:
    #         self.train_features = raw_train_features
    #     self.train_features = self.datacheck.check_data(self.train_features)
    #
    #     if baseline_type is not None:
    #         self.predictor = self._fit_baseline_model(self.train_features, train_target, baseline_type)
    #     else:
    #         self.predictor = self._fit_model(self.train_features, train_target)

    def __predict_abstraction(self,
                              test_features: Union[np.ndarray, pd.DataFrame],
                              mode: str = 'labels'):

        # if self.generator_runner is not None:
        if self.test_features is None:
            self.test_features = self.generator_runner.extract_features(ts_frame=test_features,
                                                                        dataset_name=self.dataset_name)
            self.test_features = self.datacheck.check_data(self.test_features)
        # else:
        #     self.test_features = test_features

        if isinstance(self.predictor, Pipeline):
            self.input_test_data = array_to_input_data(features_array=self.test_features, target_array=None)
            prediction_label = self.predictor.predict(self.input_test_data, output_mode=mode).predict
            return prediction_label

        else:
            if mode == 'labels':
                prediction = self.predictor.predict(self.test_features)
            else:
                prediction = self.predictor.predict_proba(self.test_features)
            return prediction
