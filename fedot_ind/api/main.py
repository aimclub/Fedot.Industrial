import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pymonad.either import Either
from sklearn import model_selection as skms
from sklearn.calibration import CalibratedClassifierCV

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.api.utils.industrial_strategy import IndustrialStrategy
from fedot_ind.api.utils.path_lib import DEFAULT_PATH_RESULTS as default_path_to_save_results
from fedot_ind.core.architecture.abstraction.decorators import DaskServer
from fedot_ind.core.architecture.pipelines.classification import SklearnCompatibleClassifier
from fedot_ind.core.architecture.preprocessing.data_convertor import ApiConverter
from fedot_ind.core.architecture.settings.computational import BackendMethods
from fedot_ind.core.operation.transformation.splitter import TSTransformer
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.constanst_repository import \
    FEDOT_GET_METRICS, FEDOT_TUNING_METRICS, \
    FEDOT_API_PARAMS, FEDOT_TUNER_STRATEGY, fedot_init_assumptions
from fedot_ind.core.repository.industrial_implementations.abstract import build_tuner
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation
from fedot_ind.tools.explain.explain import PointExplainer, RecurrenceExplainer
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


            industrial = FedotIndustrial(problem='ts_classification',
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

        # init Fedot and Industrial hyperparams and path to results
        self.output_folder = kwargs.get('output_folder', None)
        self.industrial_strategy_params = kwargs.get(
            'industrial_strategy_params', {})
        self.industrial_strategy = kwargs.get('industrial_strategy', None)
        self.path_to_composition_results = kwargs.get('history_dir', None)
        self.backend_method = kwargs.get('backend', 'cpu')
        self.task_params = kwargs.get('task_params', {})

        # TODO: unused params
        # self.model_params = kwargs.get('model_params', None)
        # self.RAF_workers = kwargs.get('RAF_workers', None)

        # create dirs with results
        if self.path_to_composition_results is None:
            prefix = './composition_results'
        else:
            prefix = self.path_to_composition_results

        Path(prefix).mkdir(parents=True, exist_ok=True)

        # create dirs with results
        if self.output_folder is None:
            self.output_folder = default_path_to_save_results
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        else:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            del kwargs['output_folder']
        # init logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    Path(
                        self.output_folder) /
                    'log.log'),
                logging.StreamHandler()])
        super(Fedot, self).__init__()

        # init hidden state variables
        self.logger = logging.getLogger('FedotIndustrialAPI')
        self.explain_methods = {'point': PointExplainer,
                                'recurrence': RecurrenceExplainer,
                                'shap': NotImplementedError,
                                'lime': NotImplementedError}
        self.solver = None
        self.predicted_labels = None
        self.predicted_probs = None
        self.predict_data = None
        self.target_encoder = None
        self.is_finetuned = False

        # map Fedot params to Industrial params
        self.config_dict = kwargs
        self.config_dict['history_dir'] = prefix
        self.config_dict['available_operations'] = kwargs.get(
            'available_operations',
            default_industrial_availiable_operation(
                self.config_dict['problem'])
        )
        self.config_dict['cv_folds'] = kwargs.get('cv_folds', 3)
        self.config_dict['optimizer'] = kwargs.get('optimizer', IndustrialEvoOptimizer)
        self.config_dict['initial_assumption'] = kwargs.get('initial_assumption', None)
        if self.config_dict['initial_assumption'] is None:
            self.config_dict['initial_assumption'] = Either(value=self.industrial_strategy,
                                                            monoid=[self.config_dict['problem'],
                                                                    self.industrial_strategy == 'anomaly_detection']). \
                either(left_function=fedot_init_assumptions,
                       right_function=fedot_init_assumptions)

        self.config_dict['use_input_preprocessing'] = kwargs.get(
            'use_input_preprocessing', False)

        if self.task_params is not None and self.config_dict['problem'] == 'ts_forecasting':
            self.config_dict['task_params'] = TsForecastingParams(
                forecast_length=self.task_params['forecast_length'])

        # create API subclasses for side task
        self.__init_experiment_setup()
        self.condition_check = ApiConverter()
        self.industrial_strategy_class = IndustrialStrategy(
            api_config=self.config_dict,
            industrial_strategy=self.industrial_strategy,
            industrial_strategy_params=self.industrial_strategy_params,
            logger=self.logger)
        self.industrial_strategy = self.industrial_strategy \
            if self.industrial_strategy != 'anomaly_detection' else None

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')
        # industrial_params = [p for p in self.config_dict.keys() if p not in list(FEDOT_API_PARAMS.keys())]
        # [self.config_dict.pop(x, None) for x in industrial_params]

        industrial_params = set(self.config_dict.keys()) - \
            set(FEDOT_API_PARAMS.keys())
        for param in industrial_params:
            self.config_dict.pop(param, None)

        backend_method_current, backend_scipy_current = BackendMethods(
            self.backend_method).backend
        globals()['backend_methods'] = backend_method_current
        globals()['backend_scipy'] = backend_scipy_current

    def __init_solver(self):
        self.logger.info('Initialising Industrial Repository')
        self.repo = IndustrialModels().setup_repository()
        self.logger.info('Initialising Dask Server')
        self.config_dict['initial_assumption'] = self.config_dict['initial_assumption'].build()
        self.dask_client = DaskServer().client
        self.logger.info(
            f'LinK Dask Server - {self.dask_client.dashboard_link}')
        self.logger.info('Initialising solver')
        self.solver = Fedot(**self.config_dict)

    def shutdown(self):
        self.dask_client.close()
        del self.dask_client

    def fit(self,
            input_data: tuple,
            **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """
        self.train_data = deepcopy(
            input_data)  # we do not want to make inplace changes
        input_preproc = DataCheck(
            input_data=self.train_data,
            task=self.config_dict['problem'],
            task_params=self.task_params,
            fit_stage=True,
            industrial_task_params=self.industrial_strategy_params)
        self.train_data = input_preproc.check_input_data()
        self.target_encoder = input_preproc.get_target_encoder()
        self.__init_solver()
        custom_fit = all([self.industrial_strategy is not None, self.industrial_strategy != 'anomaly_detection'])
        fit_function = Either(value=self.train_data,
                              monoid=[self.train_data,
                                      custom_fit]
                              ).either(left_function=self.solver.fit,
                                       right_function=self.industrial_strategy_class.fit)
        self.is_finetuned = False

    def __calibrate_probs(self, industrial_model):
        model_sklearn = SklearnCompatibleClassifier(industrial_model)
        train_idx, test_idx = skms.train_test_split(self.train_data.idx,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
        X_train, y_train = self.train_data.features[train_idx, :, :], self.train_data.target[train_idx]
        X_val, y_val = self.train_data.features[test_idx, :, :], self.train_data.target[test_idx]
        train_data_for_calibration = (X_train, y_train)
        val_data = (X_val, y_val)
        model_sklearn.fit(train_data_for_calibration[0], train_data_for_calibration[1])
        cal_clf = CalibratedClassifierCV(model_sklearn, method="sigmoid", cv="prefit")
        cal_clf.fit(val_data[0], val_data[1])
        # calibrated prediction
        calibrated_proba = cal_clf.predict_proba(self.predict_data.features)
        return calibrated_proba

    def __predict_for_ensemble(self):
        predict = self.industrial_strategy_class.predict(
            self.predict_data, 'probs')
        ensemble_strat = self.industrial_strategy_class.ensemble_strategy
        predict = {strategy: np.argmax(self.industrial_strategy_class.ensemble_predictions(predict, strategy), axis=1)
                   for strategy in ensemble_strat}
        return predict

    def __abstract_predict(self, predict_mode):
        have_encoder = self.condition_check.solver_have_target_encoder(self.target_encoder)
        labels_output = predict_mode in ['labels']
        default_fedot_strategy = self.industrial_strategy is None
        custom_predict = self.solver.predict if default_fedot_strategy else self.industrial_strategy_class.predict

        predict_function = Either(value=custom_predict,
                                  monoid=['prob', labels_output]).either(
            left_function=lambda prob_func: self.solver.predict_proba,
            right_function=lambda label_func: label_func)

        def _inverse_encoder_transform(predict):
            predicted_labels = self.target_encoder.inverse_transform(
                predict)
            self.predict_data.target = self.target_encoder.inverse_transform(
                self.predict_data.target)
            return predicted_labels

        predict = Either(
            value=self.predict_data, monoid=[False, True]).then(
            function=lambda x: predict_function(x, predict_mode)).then(
            lambda x: _inverse_encoder_transform(x) if have_encoder else x).value
        return predict

    def predict(self,
                predict_data: tuple,
                predict_mode: str = 'labels',
                **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_mode: ``default='default'``. Defines the mode of prediction. Could be 'default' or 'probs'.
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.predict_data = deepcopy(predict_data)  # we do not want to make inplace changes
        self.predict_data = DataCheck(
            input_data=self.predict_data,
            task=self.config_dict['problem'],
            task_params=self.task_params,
            industrial_task_params=self.industrial_strategy_params).check_input_data()
        self.predicted_labels = self.__abstract_predict(predict_mode)
        return self.predicted_labels

    def predict_proba(self,
                      predict_data: tuple,
                      predict_mode: str = 'probs',
                      calibrate_probs: bool = False,
                      **kwargs):
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            predict_mode: ``default='default'``. Defines the mode of prediction. Could be 'default' or 'probs'.
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction probabilities
            :param calibrate_probs:

        """
        self.predict_data = deepcopy(
            predict_data)  # we do not want to make inplace changes
        self.predict_data = DataCheck(
            input_data=self.predict_data,
            task=self.config_dict['problem'],
            task_params=self.task_params,
            industrial_task_params=self.industrial_strategy_params).check_input_data()

        self.predicted_probs = self.predicted_labels if self.config_dict['problem'] \
            in ['ts_forecasting', 'regression'] \
            else self.__abstract_predict(predict_mode)
        return self.__calibrate_probs(self.solver.current_pipeline) if calibrate_probs else self.predicted_probs

    def finetune(self,
                 train_data,
                 tuning_params=None,
                 model_to_tune=None,
                 mode: str = 'head'):
        """Method to obtain prediction probabilities from trained Industrial model.

            Args:
                model_to_tune: model to fine-tune
                train_data: raw train data
                tuning_params: dictionary with tuning parameters
                mode: str, ``default='head'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

            """
        if not self.condition_check.input_data_is_fedot_type(train_data):
            input_preproc = DataCheck(input_data=train_data,
                                      task=self.config_dict['problem'],
                                      task_params=self.task_params,
                                      industrial_task_params=self.industrial_strategy_params)
            train_data = input_preproc.check_input_data()
            self.target_encoder = input_preproc.get_target_encoder()
        tuning_params = ApiConverter.tuning_params_is_none(tuning_params)
        tuned_metric = 0
        tuning_params['metric'] = FEDOT_TUNING_METRICS[self.config_dict['problem']]
        for tuner_name, tuner_type in FEDOT_TUNER_STRATEGY.items():
            if self.condition_check.solver_is_fedot_class(self.solver):
                model_to_tune = deepcopy(self.solver.current_pipeline)
            elif not self.condition_check.solver_is_none(model_to_tune):
                model_to_tune = model_to_tune
            else:
                model_to_tune = deepcopy(
                    self.config_dict['initial_assumption']).build()
            tuning_params['tuner'] = tuner_type
            pipeline_tuner, model_to_tune = build_tuner(
                self, model_to_tune, tuning_params, train_data, mode)
            if abs(pipeline_tuner.obtained_metric) > tuned_metric:
                tuned_metric = abs(pipeline_tuner.obtained_metric)
                self.solver = model_to_tune
        self.is_finetuned = True

    def _metric_evaluation_loop(self,
                                target,
                                predicted_labels,
                                predicted_probs,
                                problem,
                                metric_names,
                                rounding_order):
        valid_shape = target.shape
        if isinstance(predicted_labels, dict):
            metric_dict = {model_name: FEDOT_GET_METRICS[problem](target=target,
                                                                  metric_names=metric_names,
                                                                  rounding_order=rounding_order,
                                                                  labels=model_result,
                                                                  probs=predicted_probs) for model_name, model_result
                           in predicted_labels.items()}
            return metric_dict
        else:
            if self.condition_check.solver_have_target_encoder(
                    self.target_encoder):
                new_target = self.target_encoder.transform(target.flatten())
                labels = self.target_encoder.transform(
                    predicted_labels).reshape(valid_shape)
            else:
                new_target = target.flatten()
                labels = predicted_labels.reshape(valid_shape)

            return FEDOT_GET_METRICS[problem](target=new_target,
                                              metric_names=metric_names,
                                              rounding_order=rounding_order,
                                              labels=labels,
                                              probs=predicted_probs)

    def get_metrics(self,
                    target: Union[list, np.array] = None,
                    metric_names: tuple = ('f1', 'roc_auc', 'accuracy'),
                    rounding_order: int = 3,
                    **kwargs) -> pd.DataFrame:
        """
        Method to calculate metrics for Industrial model.

        Available metrics for classification task: 'f1', 'accuracy', 'precision', 'roc_auc', 'logloss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae', 'median_absolute_error',
        'explained_variance_score', 'max_error', 'd2_absolute_error_score', 'msle', 'mape'.

        Args:
            target: target values
            metric_names: list of metric names
            rounding_order: rounding order for metrics

        Returns:
            pandas DataFrame with calculated metrics

        """
        problem = self.config_dict['problem']
        if problem == 'classification' and self.predicted_probs is None and 'roc_auc' in metric_names:
            self.logger.info(
                'Predicted probabilities are not available. Use `predict_proba()` method first')
        if isinstance(self.predicted_probs, dict):
            metric_dict = {
                strategy: self._metric_evaluation_loop(
                    target=target,
                    problem=problem,
                    predicted_labels=self.predicted_labels[strategy],
                    predicted_probs=probs,
                    rounding_order=rounding_order,
                    metric_names=metric_names) for strategy,
                probs in self.predicted_probs.items()}

        else:
            metric_dict = self._metric_evaluation_loop(
                target=target,
                problem=problem,
                predicted_labels=self.predicted_labels,
                predicted_probs=self.predicted_probs,
                rounding_order=rounding_order,
                metric_names=metric_names)
        return metric_dict

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
            **kwargs: dictionary with metrics

        Returns:
            None

        """
        self.solver.save_metrics(**kwargs)

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """
        self.repo = IndustrialModels().setup_repository()
        dir_list = os.listdir(path)
        if not path.__contains__('pipeline_saved'):
            saved_pipe = [x for x in dir_list if x.__contains__('pipeline_saved')][0]
            path = f'{path}/{saved_pipe}'
        pipeline = Either(value=path,
                          monoid=[dir_list, 'fitted_operations' in dir_list]).either(
            left_function=lambda directory_list: [Pipeline().load(f'{path}/{p}/0_pipeline_saved') for p in
                                                  directory_list],
            right_function=lambda path: Pipeline().load(path))
        return pipeline

    def save_optimization_history(self, return_history: bool = False):
        return self.solver.history if return_history else self.solver.history.save(f"{self.output_folder}/"
                                                                                   f"optimization_history.json")

    def save_best_model(self):
        Either(value=self.solver,
               monoid=[self.solver, self.condition_check.solver_is_fedot_class(self.solver)]).either(
            left_function=lambda pipeline: pipeline.save(path=self.output_folder,
                                                         create_subdir=True,
                                                         is_datetime_in_path=True),
            right_function=lambda solver: solver.current_pipeline.save(path=self.output_folder,
                                                                       create_subdir=True,
                                                                       is_datetime_in_path=True))

    def explain(self, explaing_config: dict = {}):
        """Explain model's prediction via time series points perturbation

            Args:
                explaing_config: Additional arguments for explanation. These arguments control the
                         number of samples, window size, metric, threshold, and dataset name.
                         See the function implementation for detailed information on
                         supported arguments.
        """
        metric = explaing_config.get('metric', 'rmse')
        window = explaing_config.get('window', 5)
        samples = explaing_config.get('samples', 1)
        threshold = explaing_config.get('threshold', 90)
        name = explaing_config.get('name', 'test')
        method = explaing_config.get('method', 'point')

        explainer = self.explain_methods[method](model=self,
                                                 features=self.predict_data.features.squeeze(),
                                                 target=self.predict_data.target)

        explainer.explain(n_samples=samples, window=window, method=metric)
        explainer.visual(metric=metric, threshold=threshold, name=name)

    def return_report(self) -> pd.DataFrame:
        return self.solver.return_report() if isinstance(self.solver, Fedot) else None

    def vis_optimisation_history(self, opt_history_path: str = None,
                                 mode: str = 'all',
                                 return_history: bool = False):
        """ The function runs visualization of the composing history and the best pipeline. """
        # Gather pipeline and history.
        # matplotlib.use('TkAgg')
        history = OptHistory.load(opt_history_path + 'optimization_history.json') \
            if isinstance(opt_history_path, str) else opt_history_path
        history_visualizer = PipelineHistoryVisualizer(history)
        vis_func = {
            'fitness': (
                history_visualizer.fitness_box, dict(
                    save_path='fitness_by_generation.png', best_fraction=1)),
            'models': (
                history_visualizer.operations_animated_bar, dict(
                    save_path='operations_animated_bar.gif', show_fitness=True)),
            'diversity': (
                history_visualizer.diversity_population, dict(
                    save_path='diversity_population.gif', fps=1))}

        def plot_func(mode): return vis_func[mode][0](**vis_func[mode][1])
        Either(value=vis_func,
               monoid=[mode, mode == 'all']).either(
            left_function=plot_func,
            right_function=lambda vis_func: [func(**params) for func, params in vis_func.values()])
        return history_visualizer.history if return_history else None

    @staticmethod
    def generate_ts(ts_config: dict):
        """
        Method to generate synthetic time series

        Args:
            ts_config: dict with config for synthetic ts_data.

        Returns:
            synthetic time series data.

        """
        return TimeSeriesGenerator(ts_config).get_ts()

    @staticmethod
    def generate_anomaly_ts(ts_data,
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
        init_synth_ts, mod_synth_ts, synth_inters = generator.generate(
            time_series_data=ts_data, plot=plot, overlap=overlap)

        return init_synth_ts, mod_synth_ts, synth_inters

    @staticmethod
    def split_ts(time_series,
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

        features, target = TSTransformer().transform_for_fit(
            plot=plot, binarize=binarize, series=time_series, anomaly_dict=anomaly_dict)
        return features, target
