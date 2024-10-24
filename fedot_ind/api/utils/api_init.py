import logging
from pathlib import Path

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
        self.industrial_strategy_params = kwargs.get(
            'industrial_strategy_params', {})
        self.industrial_strategy = kwargs.get('industrial_strategy', None)
        self.path_to_composition_results = kwargs.get('history_dir', None)
        self.backend_method = kwargs.get('backend', 'cpu')
        self.task_params = kwargs.get('task_params', {})

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
        self.config_dict = kwargs
        # self.config_dict['history_dir'] = prefix
        self.preset = kwargs.get('preset', self.config_dict['problem'])
        self.config_dict['available_operations'] = kwargs.get('available_operations',
                                                              default_industrial_availiable_operation(self.preset))
        self.is_default_fedot_context = self.preset.__contains__('tabular')
        self.is_regression_task_context = self.config_dict['problem'] in ['ts_forecasting', 'regression']
        self.config_dict['cv_folds'] = kwargs.get('cv_folds', 3)
        self.config_dict['optimizer'] = kwargs.get('optimizer', IndustrialEvoOptimizer)
        self.config_dict['initial_assumption'] = kwargs.get('initial_assumption', None)
        if self.config_dict['initial_assumption'] is None:
            self.config_dict['initial_assumption'] = Either(value=self.industrial_strategy,
                                                            monoid=[self.preset,
                                                                    self.industrial_strategy == 'anomaly_detection']). \
                either(left_function=fedot_init_assumptions,
                       right_function=fedot_init_assumptions)

        self.config_dict['use_input_preprocessing'] = kwargs.get(
            'use_input_preprocessing', False)

        if self.task_params is not None and self.config_dict['problem'] == 'ts_forecasting':
            self.config_dict['task_params'] = TsForecastingParams(
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
            api_config=self.config_dict,
            industrial_strategy=self.industrial_strategy,
            industrial_strategy_params=self.industrial_strategy_params,
            logger=self.logger)
        self.industrial_strategy = self.industrial_strategy if self.industrial_strategy != 'anomaly_detection' else None

    def __init_experiment_setup(self):
        self.logger.info('Initialising experiment setup')

        industrial_params = set(self.config_dict.keys()) - \
                            set(FEDOT_API_PARAMS.keys())
        for param in industrial_params:
            self.config_dict.pop(param, None)

        # backend_method_current, backend_scipy_current = BackendMethods(
        #     self.backend_method).backend
        # globals()['backend_methods'] = backend_method_current
        # globals()['backend_scipy'] = backend_scipy_current
