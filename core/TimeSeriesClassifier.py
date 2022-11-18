import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData, array_to_input_data
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from core.api.utils.checkers_collections import DataCheck
from core.models.ExperimentRunner import ExperimentRunner
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.cv_experimenters import ClassificationExperimenter


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
                 generator_name: str,
                 generator_runner: ExperimentRunner,
                 model_hyperparams: dict,
                 ecm_model_flag: False):
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
        fedot_model = Fedot(**self.model_hyperparams)
        fedot_model.fit(features, target)
        return fedot_model

    def _fit_baseline_model(self, features: pd.DataFrame, target: np.ndarray, baseline_type: str) -> Pipeline:
        """
        Returns pipeline with the following structure:

        .. image:: img_classification_pipelines/random_forest.png
          :width: 55%

        """
        node_scaling = PrimaryNode('scaling')
        node_final = SecondaryNode('rf', nodes_from=[node_scaling])
        baseline_pipeline = Pipeline(node_final)
        input_data = array_to_input_data(features_array=features, target_array=target)
        baseline_pipeline.fit(input_data)
        return baseline_pipeline

    def fit(self, train_features: np.ndarray, train_target: np.ndarray, dataset_name: str = None,
            baseline_type: str = None) -> tuple:
        self.logger.info('START TRAINING')
        self.y_train = train_target
        self.train_features = self.generator_runner.extract_features(train_features=train_features,
                                                                     dataset_name=dataset_name)
        self.train_features = self.datacheck.check_data(self.train_features)

        if baseline_type is not None:
            self.predictor = self._fit_baseline_model(self.train_features, train_target, baseline_type)
        else:
            self.predictor = self._fit_model(self.train_features, train_target)

        return self.predictor, self.train_features

    def predict(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        self.test_features = self.generator_runner.extract_features(test_features, dataset_name)
        self.test_features = self.datacheck.check_data(self.test_features)

        if type(self.predictor) == Pipeline:
            self.input_test_data = array_to_input_data(features_array=self.test_features, target_array=None)
            prediction_label = self.predictor.predict(self.input_test_data, output_mode='labels').predict
        else:
            prediction_label = self.predictor.predict(self.test_features)
        return dict(label=prediction_label, test_features=self.test_features)

    def predict_proba(self, test_features: np.ndarray, dataset_name: str = None) -> dict:
        if self.test_features is None:
            self.test_features = self.generator_runner.extract_features(test_features, dataset_name)
            self.test_features = self.datacheck.check_data(self.test_features)

        if self.input_test_data is not None:
            prediction_proba = self.predictor.predict(self.input_test_data, output_mode='probs').predict
        else:
            prediction_proba = self.predictor.predict_proba(self.test_features)
        return dict(class_probability=prediction_proba, test_features=self.test_features)


class TimeSeriesImageClassifier(TimeSeriesClassifier):

    def __init__(self,
                 generator_name: str,
                 generator_runner: ExperimentRunner,
                 model_hyperparams: dict,
                 ecm_model_flag: False):
        super().__init__(generator_name, generator_runner, model_hyperparams, ecm_model_flag)

    def _fit_model(self, features: pd.DataFrame, target: np.ndarray) -> Fedot:
        """Fit Fedot model with feature and target.

        Args:
            features: features for training
            target: target for training

        Returns:
            Fitted Fedot model

        """
        if 'structure_optimization' not in self.model_hyperparams.keys():
            modes = {'none': {},
                     'SVD': self.model_hyperparams['svd_parameters'],
                     'SFP': self.model_hyperparams['sfp_parameters']}
            self.model_hyperparams['structure_optimization'] = self.model_hyperparams['mode']
            self.model_hyperparams['structure_optimization_params'] = modes[self.model_hyperparams['mode']]

        NN_model = ClassificationExperimenter(**self.model_hyperparams)
        NN_model.fit(num_epochs=self.model_hyperparams['epoch'])
        return NN_model
