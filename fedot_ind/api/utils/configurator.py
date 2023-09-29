import json
import logging
from enum import Enum
from typing import Union

from fedot_ind.api.utils.path_lib import PATH_TO_DEFAULT_PARAMS
from fedot_ind.core.architecture.settings.pipeline_factory import FeatureGenerator
from fedot_ind.core.models.base_extractor import BaseExtractor


class IndustrialConfigs(Enum):
    ts_classification = dict(task='ts_classification',
                             dataset=None,
                             strategy='fedot_preset',
                             model_params={'problem': 'classification',
                                           'seed': 42,
                                           'timeout': 15,
                                           'max_depth': 10,
                                           'max_arity': 4,
                                           'cv_folds': 2,
                                           'logging_level': 20,
                                           'n_jobs': -1})

    ts_regression = dict(task='ts_regression',
                         dataset=None,
                         strategy='quantile',
                         model_params={'problem': 'regression',
                                       'seed': 42,
                                       'timeout': 15,
                                       'max_depth': 10,
                                       'max_arity': 4,
                                       'cv_folds': 2,
                                       'logging_level': 20,
                                       'n_jobs': -1})

    anomaly_detection = dict(task='anomaly_detection',
                             dataset=None,
                             strategy='fedot_preset',
                             model_params={'problem': 'classification',
                                           'seed': 42})

    # image_classification = NotImplementedError
    # object_detection = NotImplementedError


class Configurator:
    """
    Class responsible for experiment configuration.

    """

    def __init__(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.experiment_dict = None

    def _base_config(self, task: str = 'ts_classification') -> dict:
        return IndustrialConfigs[task].value

    def init_experiment_setup(self, **kwargs) -> dict:
        """Initializes the experiment setup.

        Args:
            kwargs: parameters of the experiment.

        Returns:
            Dictionary with the parameters of the experiment.

        """

        self.experiment_dict = self._base_config(task=kwargs['task'])
        fedot_config = {}
        industrial_config = {
            k: v if k not in ['timeout', 'n_jobs', 'logging_level', 'metric'] else fedot_config.update({k: v})
            for k, v in kwargs.items()}
        industrial_config['output_folder'] = kwargs['output_folder']
        self.experiment_dict.update(**industrial_config)
        self.experiment_dict['model_params'].update(**fedot_config)

        self.experiment_dict['generator_class'] = self._get_generator_class()

        self.__report_experiment_setup(self.experiment_dict)

        return self.experiment_dict

    def _get_generator_class(self) -> Union[BaseExtractor, None]:
        """Support method that combines the name of the generator with the parameters from the config file.

        Returns:
            Class of the feature generator.

        """
        generator = self.experiment_dict['strategy']
        if generator is None:
            return None
        elif generator == 'fedot_preset':
            return None
        else:
            if generator.startswith('ensemble'):
                dict_of_generators = {}
                generators_to_ensemble = generator.transform_for_fit(': ')[1].transform_for_fit(' ')
                for gen in generators_to_ensemble:
                    single_gen_class = self._extract_generator_class(gen)
                    dict_of_generators[gen] = single_gen_class
                ensemble_gen_class = FeatureGenerator['ensemble'].value(list_of_generators=dict_of_generators)
                self.feature_generator = 'ensemble'
                return ensemble_gen_class

            feature_gen_class = self._extract_generator_class(generator)
            return feature_gen_class

    def _extract_generator_class(self, generator):
        feature_gen_model = FeatureGenerator[generator].value

        with open(PATH_TO_DEFAULT_PARAMS, 'r') as file:
            _feature_gen_params = json.load(file)
            feature_gen_params = _feature_gen_params[f'{generator}_extractor']

        for param in feature_gen_params:
            feature_gen_params[param] = self.experiment_dict.get(param, feature_gen_params[param])

        feature_gen_class = feature_gen_model(feature_gen_params)
        return feature_gen_class

    def __report_experiment_setup(self, experiment_dict):
        """Prints the experiment setup."""

        top_info = ['task', 'dataset', 'strategy', 'branch_nodes', 'use_cache', 'n_jobs', 'timeout']
        label, data = [], []

        for obj in top_info:
            status = experiment_dict.get(obj, None)
            if status is not None:
                label.append(obj)
                data.append(status)

        self.logger.info('Experiment setup:')
        for i in range(len(label)):
            self.logger.info(f'{label[i]} - {data[i]}')
