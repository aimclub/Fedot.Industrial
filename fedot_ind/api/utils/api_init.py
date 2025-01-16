import logging
from pathlib import Path
from typing import Union, Callable, List

from fedot.core.repository.tasks import TsForecastingParams
from joblib import cpu_count
from pymonad.either import Either

from fedot_ind.api.utils.industrial_strategy import IndustrialStrategy
from fedot_ind.core.architecture.preprocessing.data_convertor import ApiConverter
from fedot_ind.core.optimizer.FedotEvoOptimizer import FedotEvoOptimizer
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.constanst_repository import \
    FEDOT_API_PARAMS, fedot_init_assumptions, FEDOT_INDUSTRIAL_STRATEGY
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation
from fedot_ind.tools.explain.explain import PointExplainer, RecurrenceExplainer
from fedot_ind.tools.serialisation.path_lib import DEFAULT_PATH_RESULTS as default_path_to_save_results


class ConfigTemplate:
    def __init__(self):
        self.keys = {}
        self.config = {}

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            val = method(config[key]) if key in config.keys() else method()
            self.config.update({key: val})
        return self


class IndustrialConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'default_fedot_context': self.with_default_fedot_context,
                     'regression_context': self.with_regression_context,
                     'forecasting_context': self.with_forecasting_context,
                     'initial_assumption': self.with_industrial_initial_assumption,
                     'optimizer': self.with_industrial_optimizer,
                     'use_input_preprocessing': self.with_input_preprocessing,
                     'strategy_params': self.with_industrial_strategy_params
                     }
        self.explain_methods = {'point': PointExplainer,
                                'recurrence': RecurrenceExplainer,
                                'shap': NotImplementedError,
                                'lime': NotImplementedError}
        self.regression_tasks = ['ts_forecasting', 'regression']
        self.custom_industrial_strategy = FEDOT_INDUSTRIAL_STRATEGY

    def with_default_fedot_context(self, kwargs):
        self.strategy = kwargs.get('strategy', 'default')
        self.is_default_fedot_context = self.strategy.__contains__('tabular')
        return self.is_default_fedot_context

    def with_regression_context(self, kwargs):
        self.is_regression_task_context = kwargs['problem'] in self.regression_tasks
        return self.is_regression_task_context

    def with_industrial_strategy_params(self, kwargs):
        self.strategy_params = kwargs.get('strategy_params', None)
        return self.strategy_params

    def with_forecasting_context(self, kwargs):
        self.task_params = kwargs.get('task_params', {})
        is_empty_params = any([self.task_params is None, len(self.task_params) == 0])
        self.is_forecasting_context = all([not is_empty_params, kwargs['problem'] == 'ts_forecasting'])
        if self.is_forecasting_context:
            self.task_params = TsForecastingParams(forecast_length=self.task_params['forecast_length'])
        return self.is_forecasting_context

    def with_industrial_initial_assumption(self, kwargs):
        self.initial_assumption = kwargs.get('initial_assumption', None)
        problem = kwargs['problem']
        problem = problem if not self.is_default_fedot_context else f'{problem}_{self.strategy}'
        if self.initial_assumption is None:
            self.initial_assumption = Either(value=problem,
                                             monoid=[problem,
                                                     problem == 'anomaly_detection']). \
                either(left_function=fedot_init_assumptions,
                       right_function=fedot_init_assumptions)

        return self.initial_assumption

    def with_industrial_optimizer(self, kwargs):
        self.industrial_opt = kwargs.get('optimizer', IndustrialEvoOptimizer)
        return self.industrial_opt

    def with_input_preprocessing(self, kwargs):
        self.use_input_preprocessing = kwargs.get('use_input_preprocessing', False)
        return self.use_input_preprocessing

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            val = method(config)
            self.config.update({key: val})
        if self.strategy in FEDOT_INDUSTRIAL_STRATEGY:
            self.strategy = IndustrialStrategy(industrial_strategy=self.strategy,
                                               industrial_strategy_params=self.task_params,
                                               api_config=self.config)
        return self


class ComputationalConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'backend': self.with_backend,
                     'distributed': self.with_distributed,
                     'output_folder': self.with_output_folder,
                     'use_cache': self.with_cache,
                     'automl_folder': self.with_automl_folder}
        self.default_dask_params = dict(processes=False,
                                        n_workers=1,
                                        threads_per_worker=round(cpu_count() / 2),
                                        memory_limit=0.3
                                        )

    def with_backend(self, backend: str = 'cpu'):
        self.backend = backend
        return self.backend

    def with_distributed(self, distributed: dict = None):
        self.distributed = distributed if distributed is not None else self.default_dask_params
        return self.distributed

    def with_output_folder(self, output_folder: str = None):
        self.history_dir = output_folder
        return self.history_dir

    def with_cache(self, cache_dict: dict = None):
        self.cache = cache_dict
        return self.cache

    def with_automl_folder(self, automl_folder: str = None):
        self.automl_folder = automl_folder
        return self.automl_folder


class AutomlConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'task': self.with_task,
                     'task_params': self.with_task_params,
                     'initial_assumption': self.with_initial_assumption,
                     'use_automl': self.with_automl,
                     'available_operations': self.with_available_operations,
                     'optimisation_strategy': self.with_optimisation_strategy}

    def with_task(self, task: str = None):
        self.task = task
        return self.task

    def with_task_params(self, task_params: dict = None):
        self.task_params = task_params
        return self.task_params

    def with_initial_assumption(self, initial_assumption: str = None):
        self.initial_assumption = initial_assumption
        return self.initial_assumption

    def with_automl(self, use_automl: bool = False):
        self.use_automl = use_automl
        return self.use_automl

    def with_available_operations(self, available_operations: List[str] = None):
        self.available_operations = available_operations
        if self.available_operations is None:
            self.available_operations = default_industrial_availiable_operation(self.task)
        return self.available_operations

    def with_optimisation_strategy(self, optimisation_strategy: dict = None):
        self.optimisation_strategy = optimisation_strategy
        return self.optimisation_strategy


class LearningConfig(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.keys = {'learning_strategy': self.with_learning_strategy,
                     'learning_strategy_params': self.with_learning_strategy_params,
                     'optimisation_loss': self.with_loss}

    def with_learning_strategy(self, learning_strategy: str = None):
        self.learning_strategy = learning_strategy
        return self.learning_strategy

    def with_learning_strategy_params(self, learning_strategy_params: dict = None):
        self.learning_strategy_params = learning_strategy_params
        return self.learning_strategy_params

    def with_loss(self, loss: Union[Callable, str, dict] = None):
        self.quality_loss = None
        self.computational_loss = None
        self.structural_loss = None
        if isinstance(loss, dict):
            self.quality_loss = loss.get('quality_loss')
            self.computational_loss = loss.get('computational_loss')
            self.structural_loss = loss.get('structural_loss')
        elif isinstance(loss, Callable):
            self.quality_loss = loss
        return self.quality_loss


class ApiManager(ConfigTemplate):
    def __init__(self):
        super().__init__()
        self.null_state_object()
        self.logger = logging.getLogger("FedCoreAPI")
        self.keys = {'industrial_config': self.with_industrial_config,
                     'automl_config': self.with_automl_config,
                     'learning_config': self.with_learning_config,
                     'compute_config': self.with_compute_config}
        self.optimisation_agent = {"Industrial": IndustrialEvoOptimizer,
                                   'Fedot': FedotEvoOptimizer}
        self.condition_check = ApiConverter()

    def null_state_object(self):
        self.solver = None
        self.predicted_labels = None
        self.predicted_probs = None
        self.predict_data = None
        self.target_encoder = None
        self.is_finetuned = False

    def with_industrial_config(self, config: dict):
        self.industrial_config = IndustrialConfig().build(config)
        return self.industrial_config

    def with_automl_config(self, config: dict):
        self.automl_config = AutomlConfig().build(config)
        return self.automl_config

    def with_learning_config(self, config: dict):
        self.learning_config = LearningConfig().build(config)
        return self.learning_config

    def with_compute_config(self, config: dict):
        self.compute_config = ComputationalConfig().build(config)
        return self.compute_config

    def build(self, config: dict = None):
        for key, method in self.keys.items():
            if key in config.keys():
                method(config[key])
            else:
                method()
        return self


class OldApiManager:
    def __init__(self, **kwargs):
        self.null_state_object()
        self.path_object(kwargs)
        self.industrial_config_object(kwargs)
        self.industrial_api_object()

    def null_state_object(self):
        self.solver = None
        self.predicted_labels = None
        self.predicted_probs = None
        self.predict_data = None
        self.target_encoder = None
        self.is_finetuned = False

    def path_object(self, kwargs):
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
        self.logger = logging.getLogger('FedotIndustrialAPI')

    def industrial_config_object(self, kwargs):
        # map Fedot params to Industrial params
        self.config = kwargs
        # self.config['history_dir'] = prefix
        self.preset = kwargs.get('preset', self.config['problem'])
        self.config['available_operations'] = kwargs.get('available_operations',
                                                         default_industrial_availiable_operation(self.preset))
        self.is_default_fedot_context = self.preset.__contains__('tabular')
        self.is_regression_task_context = self.config['problem'] in ['ts_forecasting', 'regression']
        self.config['cv_folds'] = kwargs.get('cv_folds', 3)
        self.config['optimizer'] = kwargs.get('optimizer', IndustrialEvoOptimizer)
        self.config['initial_assumption'] = kwargs.get('initial_assumption', None)
        if self.config['initial_assumption'] is None:
            self.config['initial_assumption'] = Either(value=self.strategy_class,
                                                       monoid=[self.preset,
                                                               self.strategy_class == 'anomaly_detection']). \
                either(left_function=fedot_init_assumptions,
                       right_function=fedot_init_assumptions)

        self.config['use_input_preprocessing'] = kwargs.get(
            'use_input_preprocessing', False)

        if self.task_params is not None and self.config['problem'] == 'ts_forecasting':
            self.config['task_params'] = TsForecastingParams(
                forecast_length=self.task_params.forecast_length)
        self.__init_experiment_setup()

    def industrial_api_object(self):
        # init hidden state variables

        self.explain_methods = {'point': PointExplainer,
                                'recurrence': RecurrenceExplainer,
                                'shap': NotImplementedError,
                                'lime': NotImplementedError}

        # create API subclasses for side task
        self.condition_check = ApiConverter()
        self.industrial_strategy_class = IndustrialStrategy(
            api_config=self.config,
            industrial_strategy=self.strategy_class,
            industrial_strategy_params=self.strategy_params,
            logger=self.logger)
        self.industrial_strategy = self.strategy_class if self.strategy_class != 'anomaly_detection' else None
        threads = round(cpu_count() / 2)
        if self.is_default_fedot_context:
            self.dask_cluster_params = dict(processes=False,
                                            n_workers=1,
                                            threads_per_worker=threads,
                                            memory_limit=0.3
                                            )
        else:
            self.dask_cluster_params = dict(processes=False,
                                            n_workers=1,
                                            threads_per_worker=threads,
                                            memory_limit=0.3
                                            )

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')

        industrial_params = set(self.config.keys()) - \
            set(FEDOT_API_PARAMS.keys())
        for param in industrial_params:
            self.config.pop(param, None)
