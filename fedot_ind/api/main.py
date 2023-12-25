import logging
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.sequential import SequentialTuner
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.api.utils.path_lib import default_path_to_save_results
from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.synthetic.anomaly_generator import AnomalyGenerator
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


class FedotIndustrial(Fedot):
    """This class is used to run Fedot in industrial mode as FedotIndustrial.

    Args:
        input_config: dictionary with the parameters of the experiment.
        output_folder: path to the folder where the results will be saved.

    Example:
        First, configure experiment and instantiate FedotIndustrial class::

            from fedot_ind.api.main import FedotIndustrial
            from fedot_ind.tools.loader import DataLoader


            industrial = FedotIndustrial(task='ts_classification',
                                         dataset='ItalyPowerDemand',
                                         strategy='topological',
                                         use_cache=False,
                                         timeout=15,
                                         n_jobs=2,
                                         logging_level=20)

        Next, download data from UCR archive::

            train_data, test_data = DataLoader(dataset_name='ItalyPowerDemand').load_data()

        Finally, fit the model and get predictions::

            model = industrial.fit(train_features=train_data[0], train_target=train_data[1])
            labels = industrial.predict(test_features=test_data[0])
            probs = industrial.predict_proba(test_features=test_data[0])
            metric = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])

    """

    def __init__(self, **kwargs):
        self.output_folder = kwargs.get('output_folder')
        if self.output_folder is None:
            kwargs.setdefault('output_folder', default_path_to_save_results())
            Path(kwargs.get('output_folder', default_path_to_save_results())).mkdir(parents=True, exist_ok=True)
        else:

            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            del kwargs['output_folder']

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.output_folder) / 'log.log'),
                logging.StreamHandler()
            ]
        )

        super(Fedot, self).__init__()

        self.logger = logging.getLogger('FedotIndustrialAPI')

        self.config_dict = None
        self.preprocessing_model = None

        self.__init_experiment_setup(**kwargs)
        self.solver = self.__init_solver()

    def __init_experiment_setup(self, **kwargs):
        self.logger.info('Initialising experiment setup')
        # self.reporter.path_to_save = kwargs.get('output_folder')
        if 'problem' in kwargs.keys():
            self.config_dict = kwargs

    def __init_solver(self):
        self.logger.info('Initialising Industrial Repository')
        self.repo = IndustrialModels().setup_repository()
        self.logger.info('Initialising solver')
        solver = Fedot(**self.config_dict)
        return solver

    def _preprocessing_strategy(self, input_data):
        if input_data.features.size > 1000000:
            self.logger.info(f'Dataset size before preprocessing - {input_data.features.shape}')
            self.logger.info('PCA transformation was applied to input data due to dataset size')
            if len(input_data.features.shape) == 3:
                self.preprocessing_model = PipelineBuilder().add_node('pca', params={'n_components': 0.9}).build()
            else:
                self.preprocessing_model = PipelineBuilder().add_node('pca', params={'n_components': 0.9}).build()
            self.preprocessing_model.fit(input_data)
            self.logger.info('Dimension reduction finished')

    def fit(self, input_data, **kwargs) -> Pipeline:
        """
        Method for training Industrial model.

        Args:
            train_features: raw train data
            train_target: target values
            **kwargs: additional parameters

        Returns:
            :class:`Pipeline` object.

        """

        input_data = DataCheck(input_data=input_data, task=self.config_dict['problem']).check_input_data()
        self._preprocessing_strategy(input_data)
        if self.preprocessing_model is not None:
            input_data.features = self.preprocessing_model.predict(input_data).predict
            self.logger.info(f'Train Dataset size after preprocessing - {input_data.features.shape}')
        fitted_pipeline = self.solver.fit(input_data)
        return fitted_pipeline

    def predict(self, predict_data, **kwargs) -> np.ndarray:
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction values

        """
        self.predict_data = DataCheck(input_data=predict_data, task=self.config_dict['problem']).check_input_data()
        if self.preprocessing_model is not None:
            self.predict_data.features = self.preprocessing_model.predict(self.predict_data).predict
            self.logger.info(f'Test Dataset size after preprocessing - {self.predict_data.features.shape}')
        return self.solver.predict(self.predict_data)

    def predict_proba(self, predict_data, **kwargs) -> np.ndarray:
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction probabilities

        """
        self.predict_data = DataCheck(input_data=predict_data, task=self.config_dict['task']).check_input_data()
        if self.preprocessing_model is not None:
            self.predict_data.features = self.preprocessing_model.predict(predict_data).predict
            self.logger.info(f'Test Dataset size after preprocessing - {self.predict_data.features.shape}')
        return self.solver.predict_proba(self.predict_data)

    def finetune(self, train_data, tuning_params) -> np.ndarray:
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction probabilities

        """
        train_data = DataCheck(input_data=train_data, task=self.config_dict['problem']).check_input_data()

        metric = ClassificationMetricsEnum.accuracy
        tuning_method = partial(SequentialTuner, inverse_node_order=True)
        tuning_method = SimultaneousTuner
        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(tuning_method) \
            .with_metric(metric) \
            .with_timeout(tuning_params['tuning_timeout']) \
            .with_early_stopping_rounds(tuning_params['tuning_early_stop']) \
            .with_iterations(tuning_params['tuning_iterations']) \
            .build(train_data)
        self.current_pipeline = pipeline_tuner.tune(self.current_pipeline)
        self.current_pipeline.fit(train_data)

    def finetune_predict(self, test_data) -> np.ndarray:
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction probabilities

        """
        self.predict_data = DataCheck(input_data=test_data, task=self.config_dict['problem']).check_input_data()
        return self.current_pipeline.predict(self.predict_data, 'labels').predict

    def get_metrics(self, **kwargs) -> dict:
        """
        Method to obtain Gets quality metrics

        Args:
            target: target values
            metric_names: list of metric names desired to be calculated
            **kwargs: additional parameters

        Returns:
            the dictionary with calculated metrics

        """
        return self.solver.get_metrics(**kwargs)

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

    def save_metrics(self, **kwargs) -> None:
        """
        Method to save metrics locally in csv format

        Args:
            metrics: dictionary with calculated metrics

        Returns:
            None

        """
        self.solver.save_metrics(**kwargs)

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """
        self.current_pipeline = Pipeline(use_input_preprocessing=self.solver.params.get('use_input_preprocessing'))
        self.current_pipeline.load(path)

    def save_optimization_history(self, **kwargs):
        """Plot prediction of the model"""
        self.solver.history.save(f"{self.output_folder}/optimization_history.json")

    def save_best_model(self):
        self.solver.current_pipeline.show(save_path=f'{self.output_folder}/best_model.png')

    def plot_fitness_by_generation(self, **kwargs):
        """Plot prediction of the model"""
        self.solver.history.show.fitness_box(save_path=f'{self.output_folder}/fitness_by_gen.png', best_fraction=0.5,
                                             dpi=100)

    def plot_operation_distribution(self, mode: str = 'total'):
        """Plot prediction of the model"""
        if mode == 'total':
            self.solver.history.show.operations_kde(save_path=f'{self.output_folder}/operation_kde.png', dpi=100)
        else:
            self.solver.history.show.operations_animated_bar(
                save_path=f'{self.output_folder}/history_animated_bars.gif',
                show_fitness=True, dpi=100)

    def explain(self, **kwargs):
        raise NotImplementedError()

    def generate_ts(self, ts_config: dict) -> np.array:
        """
        Method to generate synthetic time series
        Args:
            ts_config: dict with config for synthetic ts_data.

        Returns:
            synthetic time series data.

        """
        ts_generator = TimeSeriesGenerator(ts_config)
        t_series = ts_generator.get_ts()
        return t_series

    def generate_anomaly_ts(self,
                            ts_data: Union[dict, np.array],
                            anomaly_config: dict,
                            plot: bool = False,
                            overlap: float = 0.1) -> Tuple[np.array, np.array, dict]:
        """
        Method to generate anomaly time series
        Args:
            ts_data: either np.ndarray or dict with config for synthetic ts_data.
            anomaly_config: dict with config for anomaly generation
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        Returns:

        """

        generator = AnomalyGenerator(config=anomaly_config)
        init_synth_ts, mod_synth_ts, synth_inters = generator.generate(time_series_data=ts_data,
                                                                       plot=plot,
                                                                       overlap=overlap)

        return init_synth_ts, mod_synth_ts, synth_inters

    def split_ts(self, time_series: np.array,
                 anomaly_dict: dict,
                 binarize: bool = False,
                 strategy: str = 'frequent',
                 plot: bool = True) -> Tuple[np.array, np.array]:

        splitter = TSTransformer(time_series=time_series,
                                 anomaly_dict=anomaly_dict,
                                 strategy=strategy)

        train_data, test_data = splitter.transform_for_fit(plot=plot,
                                                           binarize=binarize)

        return train_data, test_data
