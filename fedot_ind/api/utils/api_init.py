import logging
from pathlib import Path
from joblib import cpu_count
from fedot.core.repository.tasks import TsForecastingParams
from pymonad.either import Either

from fedot_ind.api.utils.industrial_strategy import IndustrialStrategy
from fedot_ind.api.utils.path_lib import DEFAULT_PATH_RESULTS as default_path_to_save_results
from fedot_ind.core.architecture.preprocessing.data_convertor import ApiConverter
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.constanst_repository import \
    FEDOT_API_PARAMS, fedot_init_assumptions
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation
from fedot_ind.tools.explain.explain import PointExplainer, RecurrenceExplainer


class ApiManager:
    def __init__(self, **kwargs):
        self.null_state_object()
        self.user_config_object(kwargs)
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

    def user_config_object(self, kwargs):
        self.output_folder = kwargs.get('output_folder', None)
        self.strategy_params = kwargs.get(
            'strategy_params', None)
        self.strategy_class = kwargs.get('strategy', None)
        self.path_to_composition_results = kwargs.get('history_dir', None)
        self.backend_method = kwargs.get('backend', 'cpu')
        self.task_params = kwargs.get('task_params', {})
        self.optimizer_params = kwargs.get('optimizer_params', None)

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
                forecast_length=self.task_params['forecast_length'])
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

        # backend_method_current, backend_scipy_current = BackendMethods(
        #     self.backend_method).backend
        # globals()['backend_methods'] = backend_method_current
        # globals()['backend_scipy'] = backend_scipy_current
