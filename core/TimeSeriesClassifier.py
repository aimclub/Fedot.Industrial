from typing import Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from numpy import ndarray

from core.models.ExperimentRunner import ExperimentRunner
from core.operation.utils.analyzer import PerformanceAnalyzer
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector


class TimeSeriesClassifier:
    """
    Class responsible for interaction with Fedot classifier.

    :param feature_generator_dict: dict with feature generators
    :param model_hyperparams: dict with hyperparams for Fedot model
    :param ecm_model_flag: bool if error correction model is used
    """

    def __init__(self,
                 # feature_generator_dict: dict,
                 generator_name: str,
                 generator_runner: ExperimentRunner,
                 model_hyperparams: dict,
                 ecm_model_flag: False):
        self.y_train = None
        self.predictor_list = None
        self.train_features = None
        # self.feature_generator_dict = feature_generator_dict
        self.generator_name = generator_name
        self.generator_runner = generator_runner
        self.feature_generator_dict = {self.generator_name: self.generator_runner}
        self.model_hyperparams = model_hyperparams
        self.ecm_model_flag = ecm_model_flag
        # self._init_composer()
        self._init_builder()

    # def _init_composer(self) -> None:
    #     """
    #     Initialize composer with all operations.
    #
    #     :return: None
    #     """
    #     self.composer = FeatureGeneratorComposer()
    #     if self.feature_generator_dict is not None:
    #         for operation_name, operation_functionality in self.feature_generator_dict.items():
    #             self.composer.add_operation(operation_name, operation_functionality)
    #
    #     self.list_of_generators = list(self.composer.dict.values())

    def _init_builder(self) -> None:
        """
        Initialize builder with all operations combining generator name and transformation method.

        :return: None
        """
        for name, runner in self.feature_generator_dict.items():
            self.feature_generator_dict[name] = FeatureBuilderSelector(name, runner).select_transformation()

    def _fit_fedot_model(self, features: pd.DataFrame, target: np.ndarray) -> Fedot:
        """
        Fit Fedot model with feature and target
        :param features: pandas.DataFrame with features
        :param target: numpy.ndarray with target
        :return: Fedot model
        """
        fedot_model = Fedot(**self.model_hyperparams)
        fedot_model.fit(features, target)
        return fedot_model

    def fit(self, train_tuple: tuple, dataset_name: str) -> dict:
        self.y_train = train_tuple[1]
        # self.train_features = FeatureList(list_of_generators=self.list_of_generators,
        #                                   data=train_tuple[0],
        #                                   dataset_name=dataset_name).create()

        self.train_features = self.generator_runner.extract_features(train_tuple[0], dataset_name)

        # self.predictor_list = PredictorList(train_labels_set=train_tuple[1],
        #                                     feature_list=self.train_features,
        #                                     operation=self._fit_fedot_model).create()

        self.predictor = self._fit_fedot_model(self.train_features, train_tuple[1])

        return dict(predictor=self.predictor, train_features=self.train_features)

    def predict(self, predictor, test_tuple, dataset_name) -> dict:
        # feature_list = FeatureList(list_of_generators=self.list_of_generators,
        #                            data=test_tuple[0],
        #                            dataset_name=dataset_name).create()
        feature_list = self.generator_runner.extract_features(test_tuple[0], dataset_name)

        # predictions_list = PredictionsList(predictor_list=predictor_list,
        #                                    feature_list=feature_list,
        #                                    operation='predictions').create()

        prediction_label = predictor.predict(feature_list)
        prediction_proba = predictor.predict_proba(feature_list)

        # predictions_proba_list = PredictionsList(predictor_list=predictor_list,
        #                                          feature_list=feature_list,
        #                                          operation='predictions_proba').create()

        metrics_dict = PerformanceAnalyzer().calculate_metrics(target=test_tuple[1],
                                                               predicted_labels=prediction_label,
                                                               predicted_probs=prediction_proba)

        # metrics_dict = MetricsDict(predictions_list, predictions_proba_list, test_tuple[1]).create()

        return dict(prediction=prediction_label, predictions_proba=prediction_proba,
                    metrics=metrics_dict, test_features=feature_list)

    def predict_on_train(self) -> Union[ndarray, ndarray]:
        prediction_proba = self.predictor.predict_proba(self.train_features)

        # predictions_proba_list_train = PredictionsList(predictor_list=self.predictor_list,
        #                                                feature_list=self.train_features,
        #                                                operation='predictions_proba').create()

        return prediction_proba
