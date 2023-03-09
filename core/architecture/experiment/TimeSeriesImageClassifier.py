import os
from typing import Tuple

import numpy as np

from core.architecture.datasets.classification_datasets import CustomClassificationDataset
from core.architecture.experiment.CVModule import ClassificationExperimenter
from core.architecture.utils.utils import default_path_to_save_results
from core.models.ExperimentRunner import ExperimentRunner
from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier


class TimeSeriesImageClassifier(TimeSeriesClassifier):

    def __init__(self,
                 generator_name: str,
                 generator_runner: ExperimentRunner,
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
                                                      mode='probs')
        prediction_proba = np.concatenate(list(prediction_proba.values()), axis=0)
        return dict(class_probability=prediction_proba, test_features=self.test_features)
