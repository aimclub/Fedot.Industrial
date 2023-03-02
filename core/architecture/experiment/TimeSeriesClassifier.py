import os
from typing import Tuple

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from core.api.utils.checkers_collections import DataCheck
from core.models.BaseExtractor import BaseExtractor
from core.architecture.datasets.classification_datasets import CustomClassificationDataset
from core.architecture.experiment.CVModule import ClassificationExperimenter
from core.architecture.preprocessing.FeatureBuilder import FeatureBuilderSelector
from core.architecture.abstraction.LoggerSingleton import Logger
from core.architecture.utils.utils import path_to_save_results


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
                 generator_runner: BaseExtractor = None,
                 model_hyperparams: dict = None,
                 ecm_model_flag: bool = False):
        self.logger = Logger().get_logger()
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None

        self.datacheck = DataCheck(logger=self.logger)
        self.generator_name = generator_name
        self.generator_runner = generator_runner
        self.feature_generator_dict = {self.generator_name: self.generator_runner}
        self.model_hyperparams = model_hyperparams
        self.ecm_model_flag = ecm_model_flag

        if self.generator_runner is not None:
            self._init_builder()

    def _init_builder(self) -> None:
        """Initialize builder with all operations combining generator name and transformation method.

        """
        for name, runner in self.feature_generator_dict.items():
            self.feature_generator_dict[name] = FeatureBuilderSelector(name, runner).select_transformation()

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

    def __fit_abstraction(self,
                          train_features: np.ndarray,
                          train_target: np.ndarray,
                          dataset_name: str = None,
                          baseline_type: str = None):
        self.logger.info('START TRAINING')
        self.y_train = train_target
        if self.generator_runner is not None:
            self.train_features = self.generator_runner.extract_features(train_features=train_features,
                                                                         dataset_name=dataset_name)
        else:
            self.train_features = train_features
        self.train_features = self.datacheck.check_data(self.train_features)

        if baseline_type is not None:
            self.predictor = self._fit_baseline_model(self.train_features, train_target, baseline_type)
        else:
            self.predictor = self._fit_model(self.train_features, train_target)

    def __predict_abstraction(self,
                              test_features: np.ndarray,
                              mode: str = 'labels',
                              dataset_name: str = None):
        if self.generator_runner is not None:
            self.test_features = self.generator_runner.extract_features(test_features, dataset_name)
        else:
            self.test_features = test_features
        self.test_features = self.datacheck.check_data(self.test_features)

        if type(self.predictor) == Pipeline:
            self.input_test_data = array_to_input_data(features_array=self.test_features, target_array=None)
            prediction_label = self.predictor.predict(self.input_test_data, output_mode=mode).predict
            return prediction_label
        else:
            if mode == 'labels':
                prediction_label = self.predictor.predict(self.test_features)
            else:
                prediction_label = self.predictor.predict_proba(self.test_features)
            return prediction_label

    def fit(self, train_features: np.ndarray,
            train_target: np.ndarray,
            dataset_name: str = None,
            baseline_type: str = None) -> tuple:
        self.__fit_abstraction(train_features, train_target, dataset_name, baseline_type)
        return self.predictor

    def predict(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        prediction_label = self.__predict_abstraction(test_features=test_features,
                                                      mode='labels',
                                                      dataset_name=dataset_name)
        return prediction_label

    def predict_proba(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        prediction_proba = self.__predict_abstraction(test_features=test_features,
                                                      mode='probs',
                                                      dataset_name=dataset_name)
        return prediction_proba


class TimeSeriesImageClassifier(TimeSeriesClassifier):

    def __init__(self,
                 generator_name: str,
                 generator_runner: BaseExtractor,
                 model_hyperparams: dict,
                 ecm_model_flag: False):
        super().__init__(generator_name, generator_runner, model_hyperparams, ecm_model_flag)

    def _init_model_param(self, target: np.ndarray) -> Tuple[int, np.ndarray]:

        num_epochs = self.model_hyperparams['epoch']
        del self.model_hyperparams['epoch']

        if 'optimization_method' in self.model_hyperparams.keys():
            modes = {'none': {},
                     'SVD': self.model_hyperparams['optimization_method']['svd_parameters'],
                     'SFP': self.model_hyperparams['optimization_method']['sfp_parameters']}
            self.model_hyperparams['structure_optimization'] = self.model_hyperparams['optimization_method']['mode']
            self.model_hyperparams['structure_optimization_params'] = modes[
                self.model_hyperparams['optimization_method']['mode']]
            del self.model_hyperparams['optimization_method']

        self.model_hyperparams['models_saving_path'] = os.path.join(path_to_save_results(), 'TSCImage',
                                                                    self.generator_name,
                                                                    '../../models')
        self.model_hyperparams['summary_path'] = os.path.join(path_to_save_results(), 'TSCImage', self.generator_name,
                                                              'runs')
        self.model_hyperparams['num_classes'] = np.unique(target).shape[0]

        if target.min() != 0:
            target = target - 1

        return num_epochs, target

    def _fit_model(self, features: np.ndarray, target: np.ndarray) -> ClassificationExperimenter:
        """Fit Fedot model with feature and target.

        Args:
            features: features for training
            target: target for training

        Returns:
            Fitted Fedot model

        """
        num_epochs, target = self._init_model_param(target)

        train_dataset = CustomClassificationDataset(images=features, targets=target)
        NN_model = ClassificationExperimenter(train_dataset=train_dataset,
                                              val_dataset=train_dataset,
                                              **self.model_hyperparams)
        NN_model.fit(num_epochs=num_epochs)
        return NN_model

    def predict(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        prediction_label = self.__predict_abstraction(test_features, dataset_name)
        prediction_label = list(prediction_label.values())
        return dict(label=prediction_label, test_features=self.test_features)

    def predict_proba(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        prediction_proba = self.__predict_abstraction(test_features=test_features,
                                                      mode='probs',
                                                      dataset_name=dataset_name)
        prediction_proba = np.concatenate(list(prediction_proba.values()), axis=0)
        return dict(class_probability=prediction_proba, test_features=self.test_features)
