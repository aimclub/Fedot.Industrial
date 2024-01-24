import logging
import warnings
from pathlib import Path

import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.api.utils.path_lib import DEFAULT_PATH_RESULTS as default_path_to_save_results
from fedot_ind.core.architecture.abstraction.decorators import DaskServer
from fedot_ind.core.architecture.settings.computational import BackendMethods
from fedot_ind.core.ensemble.random_automl_forest import RAFensembler
from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.constanst_repository import BATCH_SIZE_FOR_FEDOT_WORKER, FEDOT_ASSUMPTIONS, \
    FEDOT_GET_METRICS, FEDOT_HEAD_ENSEMBLE, FEDOT_TUNING_METRICS, FEDOT_WORKER_NUM, FEDOT_WORKER_TIMEOUT_PARTITION
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation
from fedot_ind.tools.explain.explain import PointExplainer
from fedot_ind.tools.synthetic.anomaly_generator import AnomalyGenerator
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator

warnings.filterwarnings("ignore")


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
        self.preprocessing = kwargs.get('industrial_preprocessing', False)
        self.backend_method = kwargs.get('backend', 'cpu')

        if self.output_folder is None:
            self.output_folder = default_path_to_save_results
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
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

        self.solver = None
        self.predicted_labels = None
        self.predicted_probs = None
        self.predict_data = None
        self.config_dict = None
        self.ensemble_solver = None
        self.config_dict = kwargs
        self.__init_experiment_setup()

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')
        if 'industrial_preprocessing' in self.config_dict.keys():
            del self.config_dict['industrial_preprocessing']
        backend_method_current, backend_scipy_current = BackendMethods(self.backend_method).backend
        globals()['backend_methods'] = backend_method_current
        globals()['backend_scipy'] = backend_scipy_current
        self.config_dict['available_operations'] = default_industrial_availiable_operation(self.config_dict['problem'])

        self.config_dict['optimizer'] = IndustrialEvoOptimizer

    def __init_solver(self):
        self.logger.info('Initialising Industrial Repository')
        self.repo = IndustrialModels().setup_repository()
        if type(self.config_dict['available_operations']) is not list:
            solver = self.config_dict['available_operations'].build()
        else:
            self.logger.info('Initialising Dask Server')
            self.dask_client = DaskServer().client
            self.logger.info(f'LinK Dask Server - {self.dask_client.dashboard_link}')
            self.logger.info('Initialising solver')
            self.config_dict['initial_assumption'] = FEDOT_ASSUMPTIONS[self.config_dict['problem']].build()
            solver = Fedot(**self.config_dict)
            self.repo = IndustrialModels().setup_repository()
        return solver

    def shutdown(self):
        self.dask_client.close()
        del self.dask_client

    def _preprocessing_strategy(self, input_data):
        if input_data.features.shape[0] > BATCH_SIZE_FOR_FEDOT_WORKER:
            self.logger.info('RAF algorithm was applied')
            batch_size = round(input_data.features.shape[0] / FEDOT_WORKER_NUM)
            batch_timeout = round(self.config_dict['timeout'] / FEDOT_WORKER_TIMEOUT_PARTITION)
            self.config_dict['timeout'] = batch_timeout
            self.logger.info(f'Batch_size - {batch_size}. Number of batches - 5')
            self.ensemble_solver = RAFensembler(composing_params=self.config_dict, batch_size=batch_size)
            self.logger.info(f'Number of AutoMl models in ensemble - {self.ensemble_solver.n_splits}')
            self.ensemble_solver.fit(input_data)
            self.solver = self.ensemble_solver
        else:
            self.preprocessing = False

    def fit(self,
            input_data,
            **kwargs) -> Pipeline:
        """
        Method for training Industrial model.

        Args:
            train_features: raw train data
            train_target: target values
            **kwargs: additional parameters

        Returns:
            :param input_data:
            :class:`Pipeline` object.

        """

        input_data = DataCheck(input_data=input_data, task=self.config_dict['problem']).check_input_data()
        self.solver = self.__init_solver()
        if self.preprocessing:
            self._preprocessing_strategy(input_data)
            fitted_pipeline = self.ensemble_solver
        else:
            fitted_pipeline = self.solver.fit(input_data)
        return fitted_pipeline

    def predict(self,
                predict_data,
                **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction values
            :param predict_data:

        """
        self.predict_data = DataCheck(input_data=predict_data, task=self.config_dict['problem']).check_input_data()
        predict = self.solver.predict(self.predict_data)
        self.predicted_labels = predict if isinstance(self.solver, Fedot) else predict.predict
        return self.predicted_labels

    def predict_proba(self,
                      predict_data,
                      **kwargs):
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            test_features: raw test data

        Returns:
            the array with prediction probabilities
            :param predict_data:

        """
        self.predict_data = DataCheck(input_data=predict_data, task=self.config_dict['problem']).check_input_data()
        probs = self.solver.predict_proba(self.predict_data)
        self.predicted_probs = probs if isinstance(self.solver, Fedot) else probs.predict_proba
        return self.predicted_probs

    def finetune(self,
                 train_data,
                 tuning_params=None,
                 mode: str = 'full'):
        """
            Method to obtain prediction probabilities from trained Industrial model.

            Args:
                train_data: raw train data
                tuning_params: dictionary with tuning parameters
                mode: str, ``default='full'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

            """

        train_data = DataCheck(input_data=train_data, task=self.config_dict['problem']).check_input_data()
        if tuning_params is None:
            tuning_params = {}
        metric = FEDOT_TUNING_METRICS[self.config_dict['problem']]
        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(metric) \
            .with_timeout(tuning_params.get('tuning_timeout', 2)) \
            .with_early_stopping_rounds(tuning_params.get('tuning_early_stop', 5)) \
            .with_iterations(tuning_params.get('tuning_iterations', 10)) \
            .build(train_data)
        if mode == 'full':
            batch_pipelines = [automl_branch for automl_branch in self.solver.current_pipeline.nodes if
                               automl_branch.name in FEDOT_HEAD_ENSEMBLE]
            for b_pipeline in batch_pipelines:
                b_pipeline.fitted_operation.current_pipeline = pipeline_tuner.tune(
                    b_pipeline.fitted_operation.current_pipeline)
                b_pipeline.fitted_operation.current_pipeline.fit(train_data)
        pipeline_tuner.tune(self.solver.current_pipeline)
        self.solver.current_pipeline.fit(train_data)

    def get_metrics(self, target=None,
                    metric_names=None,
                    rounding_order=3,
                    **kwargs) -> pd.DataFrame:
        """
        Method to calculate metrics for Industrial model.

        Available metrics for classification task: 'f1', 'accuracy', 'precision', 'roc_auc', 'log_loss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae', 'median_absolute_error',
        'explained_variance_score', 'max_error', 'd2_absolute_error_score', 'msle', 'mape'.

        Args:
            target (np.ndarray): target values
            metric_names (list): list of metric names
            rounding_order (int): rounding order for metrics

        Returns:
            pandas DataFrame with calculated metrics

        """
        return FEDOT_GET_METRICS[self.config_dict['problem']](target=target,
                                                              metric_names=metric_names,
                                                              rounding_order=rounding_order,
                                                              labels=self.predicted_labels,
                                                              probs=self.predicted_probs)

    def save_predict(self, predicted_data, **kwargs) -> None:
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
        try:
            return self.solver.current_pipeline.show(save_path=f'{self.output_folder}/best_model.png')
        except Exception:
            return self.current_pipeline.show(save_path=f'{self.output_folder}/best_model.png')

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
        """ Explain model's prediction via time series points perturbation

        Args:
            samples: int, ``default=1``. Number of samples to explain.
            window: int, ``default=5``. Window size for perturbation.
            metric: str ``default='rmse'``. Distance metric for perturbation impact assessment.
            threshold: int, ``default=90``. Threshold for perturbation impact assessment.
            name: str, ``default='test'``. Name of the dataset to be placed on plot.

        """
        methods = {'point': PointExplainer,
                   'shap': NotImplementedError,
                   'lime': NotImplementedError}

        explainer = methods[kwargs.get('method', 'point')](model=self.solver,
                                                           features=self.predict_data.features,
                                                           target=self.predict_data.target)
        metric = kwargs.get('metric', 'rmse')
        window = kwargs.get('window', 5)
        samples = kwargs.get('samples', 1)
        threshold = kwargs.get('threshold', 90)
        name = kwargs.get('name', 'test')

        explainer.explain(n_samples=samples, window=window, method=metric)
        explainer.visual(threshold=threshold, name=name)

    def generate_ts(self, ts_config: dict):
        """
        Method to generate synthetic time series

        Args:
            ts_config: dict with config for synthetic ts_data.

        Returns:
            synthetic time series data.

        """
        return TimeSeriesGenerator(ts_config).get_ts()

    def generate_anomaly_ts(self,
                            ts_data,
                            anomaly_config: dict,
                            plot: bool = False,
                            overlap: float = 0.1):
        """
        Method to generate anomaly time series

        Args:
            ts_data: either np.ndarray or dict with config for synthetic ts_data.
            anomaly_config: dict with config for anomaly generation
            overlap: float, ``default=0.1``. Defines the maximum overlap between anomalies.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            returns initial time series data, modified time series data and dict with anomaly intervals.

        """

        generator = AnomalyGenerator(config=anomaly_config)
        init_synth_ts, mod_synth_ts, synth_inters = generator.generate(time_series_data=ts_data,
                                                                       plot=plot,
                                                                       overlap=overlap)

        return init_synth_ts, mod_synth_ts, synth_inters

    def split_ts(self, time_series,
                 anomaly_dict: dict,
                 binarize: bool = False,
                 plot: bool = True) -> tuple:
        """
        Method to split time series with anomalies into features and target.

        Args:
            time_series (npp.array):
            anomaly_dict (dict): dictionary with anomaly labels as keys and anomaly intervals as values.
            binarize: if True, target will be binarized. Recommended for classification task if classes are imbalanced.
            plot: if True, plot initial and modified time series data with rectangle spans of anomalies.

        Returns:
            features (pd.DataFrame) and target (np.array).

        """

        features, target = TSTransformer().transform_for_fit(plot=plot,
                                                             binarize=binarize,
                                                             series=time_series,
                                                             anomaly_dict=anomaly_dict)
        return features, target
