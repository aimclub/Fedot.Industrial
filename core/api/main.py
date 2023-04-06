import logging
from typing import List, Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline

from core.architecture.settings.task_factory import TaskGenerator
from core.api.utils.reader_collections import DataReader, YamlReader
from core.api.utils.reporter import ReporterTSC
from core.architecture.utils.utils import default_path_to_save_results


class FedotIndustrial(Fedot):
    """
    This class is used to run Fedot in industrial mode as FedotIndustrial.

    Args:

    """

    def __init__(self,
                 input_config: Union[dict, str] = None,
                 output_folder: str = None):
        super(Fedot, self).__init__()

        self.logger = logging.getLogger('FedotIndustrialAPI')

        self.reporter = ReporterTSC()
        self.solvers = {method.name: method.value for method in TaskGenerator}
        self.YAML = YamlReader()
        self.reader = DataReader()

        self.input_config = input_config
        self.config_dict = None
        self.output_folder = output_folder

        self.__init_experiment_setup()
        self.solver = self.__init_solver()

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')
        if not self.output_folder:
            self.output_folder = default_path_to_save_results()
        self.reporter.path_to_save = self.output_folder

        self.config_dict = self.YAML.init_experiment_setup(self.input_config)

    def __init_solver(self):
        self.logger.info('Initialising solver')
        solver_params = dict(generator_name=self.config_dict['feature_generator'],
                             generator_runner=self.config_dict['generator_class'],
                             model_hyperparams=self.config_dict['model_params'],
                             ecm_model_flag=self.config_dict['error_correction'],
                             dataset_name=self.config_dict['dataset'],
                             output_dir=self.output_folder)

        if self.config_dict['feature_generator'] is None and self.config_dict['task'] == 'ts_classification':
            solver = TaskGenerator[self.config_dict['task']].value[1]
        else:
            solver = TaskGenerator[self.config_dict['task']].value[0]
        return solver(solver_params)

    def fit(self,
            train_features: pd.DataFrame,
            train_target: np.ndarray,
            **kwargs) -> Pipeline:
        """
        Method for training Industrial model.

        Args:
            train_features: raw time series data
            train_target: target labels
            kwargs: additional parameters for solver, for example `baseline_type` model type that could be selected
                    instead of Fedot pipeline

        Returns:
            :class:`Pipeline` object.

        """

        fitted_pipeline = self.solver.fit(train_ts_frame=train_features,
                                          train_target=train_target,
                                          **kwargs)
        return fitted_pipeline

    def predict(self,
                test_features: pd.DataFrame,
                ) -> np.ndarray:
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction values

        """
        return self.solver.predict(test_features=test_features)

    def predict_proba(self,
                      test_features: pd.DataFrame) -> np.ndarray:
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction probabilities

        """
        return self.solver.predict_proba(test_features=test_features)

    def get_metrics(self,
                    target: Union[np.ndarray, pd.Series] = None,
                    metric_names: Union[str, List[str]] = ('f1', 'roc_auc', 'accuracy', 'logloss', 'precision'),
                    **kwargs) -> dict:
        """
        Method to obtain Gets quality metrics

        Args:
            target: target values
            metric_names: list of metric names desired to be calculated
            **kwargs: additional parameters

        Returns:
            the dictionary with calculated metrics

        """
        return self.solver.get_metrics(target, metric_names)

    def save_predict(self, predicted_data: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """
        Method to save prediction locally in csv format

        Args:
            predicted_data: predicted data. For TSC task it could be either labels or probabilities

        Returns:
            None

        """
        kind = kwargs.get('kind')
        self.solver.save_prediction(predicted_data, kind=kind)

    def save_metrics(self, metrics: dict) -> None:
        """
        Method to save metrics locally in csv format

        Args:
            metrics: dictionary with calculated metrics

        Returns:
            None
        """
        self.solver.save_metrics(metrics)

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model
        """
        # self.pipeline.load(path)
        raise NotImplementedError()

    def plot_prediction(self, **kwargs):
        """Plot prediction of the model"""
        raise NotImplementedError()

    def explain(self, **kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    datasets = ['ItalyPowerDemand'
                ]

    for dataset_name in datasets:
        config = dict(task='ts_classification',
                      dataset=dataset_name,
                      feature_generator='statistical',
                      use_cache=False,
                      error_correction=False,
                      timeout=1,
                      n_jobs=2,
                      window_sizes='auto')

        industrial = FedotIndustrial(input_config=config, output_folder=None)
        train_data, test_data, _ = industrial.reader.read(dataset_name=dataset_name)
        model = industrial.fit(train_features=train_data[0], train_target=train_data[1])

        labels = industrial.predict(test_features=test_data[0])
        probs = industrial.predict_proba(test_features=test_data[0])
        metric = industrial.get_metrics(target=test_data[1],
                                        metric_names=['f1', 'roc_auc'])

        for pred, kind in zip([labels, probs], ['labels', 'probs']):
            industrial.save_predict(predicted_data=pred, kind=kind)

        industrial.save_metrics(metrics=metric)
