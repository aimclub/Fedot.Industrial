import numpy as np
import pandas as pd
from fedot.api.main import Fedot

from core.operation.utils.Composer import FeatureGeneratorComposer
from core.operation.utils.FeatureBuilder import FeatureBuilderSelector
from core.operation.utils.TSDatatypes import FeatureList, MetricsDict, PredictionsList, PredictorList


class TimeSeriesClassifier:
    """
    Class responsible for interaction with Fedot classifier
    """
    def __init__(self,
                 feature_generator_dict: dict,
                 model_hyperparams: dict,
                 error_correction: False):
        self.feature_generator_dict = feature_generator_dict
        self.model_hyperparams = model_hyperparams
        self.error_correction = error_correction
        self._init_composer()
        self._init_builder()

    def _init_composer(self) -> None:
        """
        Initialize composer with all operations
        :return:
        """
        self.composer = FeatureGeneratorComposer()
        if self.feature_generator_dict is not None:
            for operation_name, operation_functionality in self.feature_generator_dict.items():
                self.composer.add_operation(operation_name, operation_functionality)

        self.list_of_generators = list(self.composer.dict.values())

    def _init_builder(self) -> None:
        """
        Initialize builder with all operations combining generator name and transformation method
        :return:
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

        feature_list = FeatureList(list_of_generators=self.list_of_generators,
                                   data=train_tuple[0],
                                   dataset_name=dataset_name).create()

        predictor_list = PredictorList(train_tuple[1],
                                       feature_list,
                                       self._fit_fedot_model).create()

        return dict(predictors=predictor_list, train_features=feature_list)

    def predict(self, predictor_list, test_tuple, dataset_name) -> dict:

        feature_list = FeatureList(list_of_generators=self.list_of_generators,
                                   data=test_tuple[0],
                                   dataset_name=dataset_name).create()

        predictions_list = PredictionsList(predictor_list=predictor_list,
                                           feature_list=feature_list,
                                           operation='predictions').create()

        predictions_proba_list = PredictionsList(predictor_list=predictor_list,
                                                 feature_list=feature_list,
                                                 operation='predictions_proba').create()

        metrics_dict = MetricsDict(predictions_list, predictions_proba_list, test_tuple[1]).create()

        return dict(prediction=predictions_list,
                    prediction_proba=predictions_proba_list,
                    metrics=metrics_dict,
                    test_features=feature_list)
