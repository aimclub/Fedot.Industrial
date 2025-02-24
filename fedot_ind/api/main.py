import os
import warnings
from copy import deepcopy
from functools import partial
from typing import Union, Callable, Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import OutputData, InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pymonad.either import Either
from pymonad.tools import curry
from sklearn import model_selection as skms
from sklearn.calibration import CalibratedClassifierCV

from fedot_ind.api.utils.api_init import ApiManager
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.architecture.abstraction.decorators import DaskServer, exception_handler
from fedot_ind.core.architecture.pipelines.classification import (
    SklearnCompatibleClassifier,
)
from fedot_ind.core.repository.constanst_repository import (
    FEDOT_GET_METRICS,
    FEDOT_TUNER_STRATEGY,
    FEDOT_TUNING_METRICS
)
from fedot_ind.core.repository.industrial_implementations.abstract import build_tuner
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

warnings.filterwarnings("ignore")


class FedotIndustrial(Fedot):
    """Main class for Industrial API. It provides a high-level interface for working with the
    Fedot framework. The class allows you to train, predict, and evaluate models for time series.
    All arguments are passed as keyword arguments and handled by the ApiManager class.

    Args:
        problem: str. The type of task to solve. Available options: 'ts_forecasting', 'ts_classification', 'ts_regression'.
        timeout: int. Time for model design (in minutes): ``None`` or ``-1`` means infinite time.
                logging_level: logging levels are the same as in
            `built-in logging library <https://docs.python.org/3/library/logging.html>`_.

            .. details:: Possible options:

                - ``50`` -> critical
                - ``40`` -> error
                - ``30`` -> warning
                - ``20`` -> info
                - ``10`` -> debug
                - ``0`` -> nonset
        backend_method: str. Default `cpu`. The method for backend. Available options: 'cpu', 'dask'.
        initial_assumption: Pipeline = None. The initial pipeline for the model.
        optimizer_params: dict = None.
        task_params: dict = None.
        strategy: str = None.
        strategy_params: dict = None.
        available_operations: list = None.
        output_folder: str = './output'.

    Example:
        First, configure experiment and instantiate FedotIndustrial class::

            from fedot_ind.api.main import FedotIndustrial
            from fedot_ind.tools.loader import DataLoader


            industrial = FedotIndustrial(problem='ts_classification',
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
        super(Fedot, self).__init__()
        self.manager = ApiManager().build(kwargs)
        self.logger = self.manager.logger

    def __init_industrial_backend(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Industrial Repository')
        if self.manager.industrial_config.is_default_fedot_context:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Fedot Evolutionary Optimisation params')
            self.repo = IndustrialModels().setup_default_repository()
            self.manager.automl_config.optimisation_strategy = self.manager.optimisation_agent['Fedot']
        else:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Industrial Evolutionary Optimisation params')
            self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
            optimisation_agent = self.manager.automl_config.optimisation_strategy['optimisation_agent']
            optimisation_params = self.manager.automl_config.optimisation_strategy['optimisation_strategy']
            self.manager.automl_config.optimisation_strategy = partial(
                self.manager.optimisation_agent[optimisation_agent],
                optimisation_params=optimisation_params)
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Dask Server')
        if self.manager.automl_config.config['initial_assumption'] is None:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.industrial_config.config['initial_assumption'].build()
        else:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.automl_config.config['initial_assumption'].build()
        dask_server = DaskServer(self.manager.compute_config.distributed)
        self.manager.dask_client = dask_server.client
        self.manager.dask_cluster = dask_server.cluster
        self.logger.info(f'Link Dask Server - {self.manager.dask_client.dashboard_link}')
        self.logger.info('-' * 50)
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(
            **self.manager.learning_config.config['learning_strategy_params'],
            metric=self.manager.learning_config.config['optimisation_loss'],
            problem=self.manager.automl_config.config['task'],
            task_params=self.manager.industrial_config.task_params
            if self.manager.industrial_config.is_forecasting_context else self.manager.automl_config.config
            ['task_params'], optimizer=self.manager.automl_config.optimisation_strategy,
            available_operations=self.manager.automl_config.config['available_operations'],
            initial_assumption=self.manager.automl_config.config['initial_assumption'])
        return input_data

    def _process_input_data(self, input_data):
        train_data, self.target_encoder = Either.insert(input_data).then(lambda data: deepcopy(data)). \
            then(lambda data: DataCheck(input_data=data, task=self.manager.automl_config.config['task'],
                                        task_params=self.manager.automl_config.config['task_params'], fit_stage=True,
                                        industrial_task_params=self.manager.industrial_config.strategy_params)). \
            then(lambda data_cls: (data_cls.check_input_data(), data_cls.get_target_encoder())).value
        train_data.features = train_data.features.squeeze() if self.manager.industrial_config.is_default_fedot_context \
            else train_data.features
        return train_data

    def __calibrate_probs(self, probability_model, predict_data):
        model_sklearn = SklearnCompatibleClassifier(probability_model)
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
        calibrated_proba = cal_clf.predict_proba(predict_data.features)
        return calibrated_proba

    def __predict_for_ensemble(self):
        predict = self.manager.industrial_config.strategy.predict(self.predict_data, 'probs')
        ensemble_strat = self.manager.industrial_config.strategy.ensemble_strategy
        predict = {strategy: np.argmax(self.manager.industrial_config.strategy.ensemble_predictions(
            predict, strategy), axis=1) for strategy in ensemble_strat}
        return predict

    def __abstract_predict(self, predict_data, predict_mode):
        have_encoder = self.manager.condition_check.solver_have_target_encoder(self.target_encoder)
        custom_predict = all([not self.manager.condition_check.solver_is_fedot_class(self.manager.solver),
                              not self.manager.condition_check.solver_is_pipeline_class(self.manager.solver)])

        def _inverse_encoder_transform(predict):
            predicted_labels = self.target_encoder.inverse_transform(predict)
            self.predict_data.target = self.target_encoder.inverse_transform(self.predict_data.target)
            return predicted_labels

        def predict_func(predict_from_solver):
            is_labels_output = predict_mode in ['labels']
            if self.manager.condition_check.solver_is_pipeline_class(self.manager.solver):
                predict = self.manager.solver.predict(predict_from_solver, predict_mode)
            else:
                if is_labels_output:
                    predict = self.manager.solver.predict(predict_from_solver)
                else:
                    predict = self.manager.solver.predict_proba(predict_from_solver)
            return predict

        predict = Either(value=predict_data,
                         monoid=[predict_data, custom_predict]).either(
            left_function=lambda predict_from_solver: predict_func(predict_from_solver),
            right_function=lambda predict_from_custom: self.manager.solver.predict(predict_from_custom))
        predict = Either.insert(predict).then(lambda x: _inverse_encoder_transform(x) if have_encoder else x). \
            then(lambda x: x.predict if isinstance(predict, OutputData) else x).value
        if predict_data.task.task_type.value.__contains__('forecasting'):
            predict = predict[-predict_data.task.task_params.forecast_length:]
        return predict

    def _metric_evaluation_loop(self,
                                target,
                                predicted_labels,
                                predicted_probs,
                                problem,
                                metric_names,
                                rounding_order,
                                train_data,
                                seasonality):
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
            if self.manager.condition_check.solver_have_target_encoder(self.target_encoder):
                new_target = self.target_encoder.transform(target.flatten())
                labels = self.target_encoder.transform(predicted_labels).reshape(valid_shape)
            else:
                new_target = target.flatten()
                labels = predicted_labels.reshape(valid_shape)

            return FEDOT_GET_METRICS[problem](target=new_target,
                                              metric_names=metric_names,
                                              rounding_order=rounding_order,
                                              labels=labels,
                                              probs=predicted_probs,
                                              train_data=train_data,
                                              seasonality=seasonality)

    def fit(self,
            input_data: tuple,
            **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """

        def fit_function(train_data): return \
            Either(value=train_data, monoid=[train_data,

                                             not isinstance(self.manager.industrial_config.strategy, Callable)]). \
            either(left_function=lambda data: self.manager.industrial_config.strategy.fit(data),
                   right_function=lambda data: self.manager.solver.fit(data))

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            Either.insert(self._process_input_data(input_data)). \
                then(lambda data: self.__init_industrial_backend(data)). \
                then(lambda data: self.__init_solver(data)). \
                then(fit_function)

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
        predict_func = curry(2)(lambda predict_mode, predict_data: self.__abstract_predict(predict_data, predict_mode))
        self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
        processed_input = self._process_input_data(predict_data)
        self.manager.predict_data = processed_input
        self.manager.predicted_labels = Either.insert(processed_input).then(predict_func(predict_mode)).value

        return self.manager.predicted_labels

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
            calibrate_probs: ``default=False``. If True, calibrate probabilities

        Returns:
            the array with prediction probabilities

        """
        self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
        predict_mode = predict_mode if not self.manager.industrial_config.is_regression_task_context else 'labels'
        predict_func = curry(2)(lambda predict_mode, predict_data: self.__abstract_predict(predict_data, predict_mode))
        calibrate_func = curry(3)(lambda prob_model, data_for_calib, labels:
                                  self.__calibrate_probs(prob_model, data_for_calib) if predict_mode.__contains__(
                                      'probs') else labels)
        self.manager.predicted_probs = Either. \
            insert(self._process_input_data(predict_data)). \
            then(predict_func(predict_mode)).value
        # then(calibrate_func(self.manager.solver, predict_data)).value

        return self.manager.predicted_probs

    def finetune(self,
                 train_data: Union[InputData, dict, tuple],
                 tuning_params: Optional[dict] = None,
                 model_to_tune: Optional[Pipeline] = None,
                 return_only_fitted: bool = False):
        """Method to obtain prediction probabilities from trained Industrial model.

            Args:
                model_to_tune: model to fine-tune
                train_data: raw train data
                tuning_params: dictionary with tuning parameters
                mode: str, ``default='head'``. Defines the mode of fine-tuning. Could be 'full' or 'head'.

            """

        def _fit_pipeline(data_dict):
            data_dict['model_to_tune'].fit(data_dict['train_data'])
            return data_dict

        is_fedot_datatype = self.manager.condition_check.input_data_is_fedot_type(train_data)
        tuning_params['metric'] = FEDOT_TUNING_METRICS[self.manager.automl_config.config['task']]
        tuning_params['tuner'] = FEDOT_TUNER_STRATEGY[tuning_params.get('tuner', 'sequential')]

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            model_to_tune = Either.insert(train_data). \
                then(lambda data: self._process_input_data(data) if not is_fedot_datatype else data). \
                then(lambda data: self.__init_industrial_backend(data)). \
                then(lambda processed_data: {'train_data': processed_data} |
                                            {'model_to_tune': model_to_tune.build()} |
                                            {'tuning_params': tuning_params}). \
                then(lambda dict_for_tune: _fit_pipeline(dict_for_tune)['model_to_tune'] if return_only_fitted
                     else build_tuner(self, **dict_for_tune)).value

        self.manager.is_finetuned = True
        self.manager.solver = model_to_tune

    def get_metrics(self,
                    labels: np.ndarray,
                    probs: np.ndarray,
                    target: Union[list, np.array] = None,
                    metric_names: tuple = None,
                    rounding_order: int = 3,
                    train_data: Union[list, np.array] = None,
                    seasonality: int = 1) -> pd.DataFrame:
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
        problem = self.manager.automl_config.task
        warning_about_probs = all([problem == 'classification',
                                   probs is None,
                                   'roc_auc' in metric_names])
        if warning_about_probs:
            self.logger.info('Predicted probabilities are not available. Use `predict_proba()` method first')

        self.metric_dict = self._metric_evaluation_loop(
            target=target,
            problem=problem,
            predicted_labels=labels,
            predicted_probs=probs,
            rounding_order=rounding_order,
            metric_names=metric_names,
            train_data=train_data,
            seasonality=seasonality)
        return self.metric_dict

    def save(self, mode: str = 'all', **kwargs):
        is_fedot_solver = self.manager.condition_check.solver_is_fedot_class(self.manager.solver)

        def save_model(api_manager):
            return Either(value=api_manager.solver,
                          monoid=[api_manager.solver,
                                  api_manager.condition_check.solver_is_fedot_class(
                                      api_manager.solver)]). \
                either(left_function=lambda pipeline: pipeline.save(path=api_manager.compute_config.output_folder,
                                                                    create_subdir=True, is_datetime_in_path=True),
                       right_function=lambda solver: solver.current_pipeline.save(
                           path=api_manager.compute_config.output_folder,
                           create_subdir=True,
                           is_datetime_in_path=True))

        def save_opt_hist(api_manager):
            return self.manager.solver.history.save(
                f"{self.manager.compute_config.output_folder}/optimization_history.json")

        def save_metrics(api_manager):
            return self.metric_dict.to_csv(
                f'{self.manager.compute_config.output_folder}/metrics.csv')

        def save_preds(api_manager):
            return pd.DataFrame(api_manager.predicted_labels).to_csv(
                f'{self.manager.compute_config.output_folder}/labels.csv')

        method_dict = {'metrics': save_metrics, 'model': save_model, 'opt_hist': save_opt_hist,
                       'prediction': save_preds}
        self.manager.create_folder(self.manager.compute_config.output_folder)
        if not is_fedot_solver:
            del method_dict['opt_hist']

        def save_all(api_manager):
            for method in method_dict.values():
                try:
                    method(api_manager)
                except Exception as ex:
                    self.manager.logger.info(f'Error during saving. Exception - {ex}')

        Either(value=self.manager, monoid=[self.manager, mode.__contains__('all')]). \
            either(left_function=lambda api_manager: method_dict[mode](self.manager),
                   right_function=lambda api_manager: save_all(api_manager))

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

        explainer = self.manager.industrial_config.explain_methods[method](
            model=self,
            features=self.manager.predict_data.features.squeeze(),
            target=self.manager.predict_data.target
        )

        explainer.explain(n_samples=samples, window=window, method=metric)
        explainer.visual(metric=metric, threshold=threshold, name=name)

    def return_report(self) -> pd.DataFrame:
        return self.manager.solver.return_report() if isinstance(self.manager.solver, Fedot) else None

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

        def plot_func(mode):
            return vis_func[mode][0](**vis_func[mode][1])

        Either(value=vis_func,
               monoid=[mode, mode == 'all']).either(
            left_function=plot_func,
            right_function=lambda vis_func: [func(**params) for func, params in vis_func.values()]
        )
        return history_visualizer.history if return_history else None

    def shutdown(self):
        """Shutdown Dask client"""
        if self.manager.dask_client is not None:
            self.manager.dask_client.close()
            del self.manager.dask_client
        if self.manager.dask_cluster is not None:
            self.manager.dask_cluster.close()
            del self.manager.dask_cluster
