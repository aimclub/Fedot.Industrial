import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from numpy import ndarray

from core.models.ExperimentRunner import ExperimentRunner
from core.operation.utils.analyzer import PerformanceAnalyzer
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector
from core.operation.utils.LoggerSingleton import Logger


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

    def _fit_fedot_model(self, features: pd.DataFrame, target: np.ndarray) -> Fedot:
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

    def fit(self, train_tuple: tuple, dataset_name: str) -> tuple:
        self.logger.info('START TRAINING')
        self.y_train = train_tuple[1]
        self.train_features = self.generator_runner.extract_features(train_tuple[0], dataset_name, train_tuple[1])
        self.predictor = self._fit_fedot_model(self.train_features, train_tuple[1])

        return self.predictor, self.train_features

    def predict(self, test_tuple, dataset_name) -> dict:
        features = self.generator_runner.extract_features(test_tuple[0], dataset_name)
        prediction_label = self.predictor.predict(features)
        prediction_proba = self.predictor.predict_proba(features)
        metrics_dict = PerformanceAnalyzer().calculate_metrics(target=test_tuple[1],
                                                               predicted_labels=prediction_label,
                                                               predicted_probs=prediction_proba)

        return dict(prediction=prediction_label, predictions_proba=prediction_proba,
                    metrics=metrics_dict, test_features=feature_list)

    def predict_on_validation(self,
                              validatiom_tuple: tuple,
                              dataset_name: str) -> ndarray:
        val_feature_list = self.generator_runner.extract_features(validatiom_tuple[0],
                                                                  dataset_name,
                                                                  validatiom_tuple[1])

        return self.predictor.predict_proba(val_feature_list)
    def predict_on_train(self) -> Union[ndarray, ndarray]:
        prediction_proba = self.predictor.predict_proba(self.train_features)

        return prediction_proba
