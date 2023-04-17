import logging
import os
from typing import List, Union
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fedot.api.main import Fedot
from fedot.core.data.data import array_to_input_data
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.api.utils.saver_collections import ResultSaver
from fedot_ind.core.architecture.datasets.classification_datasets import CustomClassificationDataset
from fedot_ind.core.architecture.experiment.CVModule import ClassificationExperimenter
from fedot_ind.core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from fedot_ind.core.architecture.utils.utils import default_path_to_save_results
from fedot_ind.core.models.nn.inception import InceptionTimeNetwork

TSCCLF_MODEL = {
    'inception_time': InceptionTimeNetwork
}


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
        self.generator_name = params.get('feature_generator', 'statistical')
        self.model_hyperparams = params.get('model_params')
        self.generator_runner = params.get('generator_class')
        self.dataset_name = params.get('dataset')
        self.output_folder = params.get('output_folder', None)

        self.saver = ResultSaver(dataset_name=self.dataset_name,
                                 generator_name=self.generator_name,
                                 output_dir=self.output_folder)
        self.logger = logging.getLogger('TimeSeriesClassifier')
        self.datacheck = DataCheck()

        self.prediction_label = None
        self.predictor = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.input_test_data = None

        self.logger.info('TimeSeriesClassifier initialised')

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

    def __predict_abstraction(self,
                              test_features: Union[np.ndarray, pd.DataFrame],
                              mode: str = 'labels'):
        self.logger.info(f'Predicting with {self.generator_name} generator')

        if self.test_features is None:
            self.test_features = self.generator_runner.extract_features(train_features=test_features,
                                                                        dataset_name=self.dataset_name)
            self.test_features = self.datacheck.check_data(input_data=self.test_features, return_df=True)

        if isinstance(self.predictor, Pipeline):
            self.input_test_data = array_to_input_data(features_array=self.test_features, target_array=None)
            prediction_label = self.predictor.predict(self.input_test_data, output_mode=mode).predict
            return prediction_label
        else:
            if mode == 'labels':
                prediction_label = self.predictor.predict(self.test_features)
            else:
                prediction_label = self.predictor.predict_proba(self.test_features)
            return prediction_label

    def fit(self, train_ts_frame: Union[np.ndarray, pd.DataFrame],
            train_target: np.ndarray,
            **kwargs) -> object:

        baseline_type = kwargs.get('baseline_type', None)
        self.logger.info(f'Fitting model')
        self.y_train = train_target
        if self.generator_runner is None:
            raise AttributeError('Feature generator is not defined')

        train_features = self.generator_runner.extract_features(train_features=train_ts_frame,
                                                                dataset_name=self.dataset_name)

        self.train_features = self.datacheck.check_data(input_data=train_features,
                                                        return_df=True)

        if baseline_type is not None:
            self.predictor = self._fit_baseline_model(self.train_features, train_target, baseline_type)
        else:
            self.predictor = self._fit_model(self.train_features, train_target)

        self.logger.info(
            f'Solver fitted: {self.generator_name}_extractor -> fedot_pipeline ({self.predictor.current_pipeline})')
        return self.predictor

    def predict(self, test_features: np.ndarray, **kwargs) -> dict:
        self.prediction_label = self.__predict_abstraction(test_features=test_features,
                                                           mode='labels')
        return self.prediction_label

    def predict_proba(self, test_features: np.ndarray, **kwargs) -> dict:
        self.prediction_proba = self.__predict_abstraction(test_features=test_features,
                                                           mode='probs', )
        return self.prediction_proba

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


class TimeSeriesImageClassifier(TimeSeriesClassifier):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

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

        self.model_hyperparams['models_saving_path'] = os.path.join(default_path_to_save_results(), 'TSCImage',
                                                                    self.generator_name,
                                                                    '../../models')
        self.model_hyperparams['summary_path'] = os.path.join(default_path_to_save_results(), 'TSCImage',
                                                              self.generator_name,
                                                              'runs')
        self.model_hyperparams['num_classes'] = np.unique(target).shape[0]

        if target.min() != 0:
            target = target - 1

        return num_epochs, target

    def fit(self, train_ts_frame: Union[np.ndarray, pd.DataFrame],
            train_target: np.ndarray,
            **kwargs) -> object:
        if type(train_ts_frame) is pd.DataFrame:
            train_ts_frame =train_ts_frame.values
        return self._fit_model(features=train_ts_frame, target=train_target)

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

    def predict(self, test_ts_frame: np.ndarray, dataset_name: str = None) -> dict:
        prediction_label = self.__predict_abstraction(test_ts_frame, dataset_name)
        prediction_label = list(prediction_label.values())
        return dict(label=prediction_label, test_features=self.test_features)

    def predict_proba(self, test_ts_frame: np.ndarray, dataset_name: str = None) -> dict:
        prediction_proba = self.__predict_abstraction(test_features=test_ts_frame,
                                                      mode='probs')
        prediction_proba = np.concatenate(list(prediction_proba.values()), axis=0)
        return dict(class_probability=prediction_proba, test_features=self.test_features)


class TimeSeriesClassifierNN(TimeSeriesImageClassifier):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.device = torch.device('cuda' if params.get('gpu', False) else 'cpu')
        self.model = TSCCLF_MODEL[params.get('model', 'inception_time')].network_architecture
        self.num_epochs = params.get('num_epochs', 10)

    def _init_model_param(self, target: np.ndarray) -> Tuple[int, np.ndarray]:
        self.model_hyperparams['models_saving_path'] = os.path.join(default_path_to_save_results(), 'TSCNN',
                                                                    '../../models')
        self.model_hyperparams['summary_path'] = os.path.join(default_path_to_save_results(), 'TSCNN',
                                                              'runs')
        self.model_hyperparams['num_classes'] = np.unique(target).shape[0]

        if target.min() != 0:
            target = target - 1

        return self.num_epochs, target
