# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from pandas import DataFrame

from core.operation.utils.Composer import FeatureGeneratorComposer
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector
from core.operation.utils.TSDatatypes import FeatureList, MetricsDict, PredictionsList, PredictorList


class TimeSeriesClassifier:
    """
    Class responsible for interaction with Fedot classifier.

    :param feature_generator_dict: dict with feature generators
    :param model_hyperparams: dict with hyperparams for Fedot model
    :param ecm_model_flag: bool if error correction model is used
    """

    def __init__(self,
                 feature_generator_dict: dict,
                 model_hyperparams: dict,
                 ecm_model_flag: False):
        self.y_train = None
        self.predictor_list = None
        self.train_features = None
        self.feature_generator_dict = feature_generator_dict
        self.model_hyperparams = model_hyperparams
        self.ecm_model_flag = ecm_model_flag
        self._init_composer()
        self._init_builder()

    def _init_composer(self) -> None:
        """
        Initialize composer with all operations.

        :return: None
        """
        self.composer = FeatureGeneratorComposer()
        if self.feature_generator_dict is not None:
            for operation_name, operation_functionality in self.feature_generator_dict.items():
                self.composer.add_operation(operation_name, operation_functionality)

        self.list_of_generators = list(self.composer.dict.values())

    def _init_builder(self) -> None:
        """
        Initialize builder with all operations combining generator name and transformation method.

        :return: None
        """
        for operation_name, operation_functionality in self.feature_generator_dict.items():
            self.feature_generator_dict[operation_name] = \
                FeatureBuilderSelector(operation_name, operation_functionality).select_transformation()

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
        self.train_features = FeatureList(list_of_generators=self.list_of_generators,
                                          data=train_tuple,
                                          dataset_name=dataset_name).create()

        self.predictor_list = PredictorList(train_labels_set=train_tuple[1],
                                            feature_list=self.train_features,
                                            operation=self._fit_fedot_model).create()

        return dict(predictors=self.predictor_list,
                    train_features=self.train_features)

    def predict(self, predictor_list, test_tuple, dataset_name) -> List[Dict[str, Union[DataFrame, Any]]]:

        feature_list = FeatureList(list_of_generators=self.list_of_generators,
                                   data=test_tuple,
                                   dataset_name=dataset_name).create()

        predictions_list = PredictionsList(predictor_list=predictor_list,
                                           feature_list=feature_list,
                                           operation='predictions').create()

        predictions_proba_list = PredictionsList(predictor_list=predictor_list,
                                                 feature_list=feature_list,
                                                 operation='predictions_proba').create()

        metrics_dict = MetricsDict(predictions_list, predictions_proba_list, test_tuple[1]).create()

        return [dict(prediction=predictions_list[i],
                     predictions_proba=predictions_proba_list[i],
                     metrics=metrics_dict[i],
                     test_features=feature_list[i]) for i in range(len(predictions_list))]

    def predict_on_train(self) -> List[Dict[str, Union[DataFrame, Any]]]:
        predictions_proba_list_train = PredictionsList(predictor_list=self.predictor_list,
                                                       feature_list=self.train_features,
                                                       operation='predictions_proba').create()

        return predictions_proba_list_train

    def predict_on_validation(self,
                              validatiom_dataset: tuple,
                              dataset_name: str) -> List[Dict[str, Union[DataFrame, Any]]]:

        val_feature_list = FeatureList(list_of_generators=self.list_of_generators,
                                       data=validatiom_dataset,
                                       dataset_name=dataset_name).create()

        predictions_proba_list_validation = PredictionsList(predictor_list=self.predictor_list,
                                                            feature_list=val_feature_list,
                                                            operation='predictions_proba').create()

        return predictions_proba_list_validation
