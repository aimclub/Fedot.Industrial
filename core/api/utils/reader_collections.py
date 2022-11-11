import copy
import os

import numpy as np
import yaml

from core.api.utils.checkers_collections import ParameterCheck
from core.operation.utils.load_data import DataLoader
from core.operation.utils.utils import PROJECT_PATH
from core.operation.utils.LoggerSingleton import Logger


class YamlReader:
    def __init__(self,
                 feature_generator: dict = None,
                 logger: Logger = None):

        self.feature_generator = feature_generator
        self.logger = logger
        self.experiment_check = ParameterCheck(logger=self.logger)
        self.use_cache = None

    def read_yaml_config(self, config_name: str) -> None:
        """Read yaml config from './cases/config/config_name' directory as dictionary file.

        Args:
            config_name: name of the config file.

        Returns:
            None

        """
        path = os.path.join(PROJECT_PATH, 'cases', 'config', config_name)
        with open(path, "r") as input_stream:
            self.config_dict = yaml.safe_load(input_stream)
            if 'path_to_config' in list(self.config_dict.keys()):
                config_name = self.config_dict['path_to_config']
                path = os.path.join(PROJECT_PATH, config_name)
                with open(path, "r") as input_stream:
                    config_dict_template = yaml.safe_load(input_stream)
                self.config_dict = {**config_dict_template, **self.config_dict}
                del self.config_dict['path_to_config']

            self.logger.info(f'''Experiment setup:
            datasets - {self.config_dict['datasets_list']},
            feature generators - {self.config_dict['feature_generator']},
            use_cache - {self.config_dict['use_cache']},
            error_correction - {self.config_dict['error_correction']}''')

    def init_experiment_setup(self, config_name):
        self.read_yaml_config(config_name=config_name)
        for dataset_name in self.config_dict['datasets_list']:
            train_data, _ = DataLoader(dataset_name).load_data()
            self.experiment_check.check_window_sizes(config_dict=self.config_dict,
                                                     dataset_name=dataset_name,
                                                     train_data=train_data)
        experiment_dict = copy.deepcopy(self.config_dict)
        self.use_cache = experiment_dict['use_cache']

        experiment_dict['feature_generator'].clear()
        experiment_dict['feature_generator'] = dict()

        for idx, generator in enumerate(self.config_dict['feature_generator']):
            if generator.startswith('ensemble'):
                generators = generator.split(': ')[1].split(' ')
                for gen_name in generators:
                    feature_gen_class = self.get_generator_class(experiment_dict, gen_name)
                    experiment_dict['feature_generator_params']['ensemble']['list_of_generators'].update(
                        feature_gen_class)

                ensemble_class = self.get_generator_class(experiment_dict, 'ensemble')
                experiment_dict['feature_generator'].update(ensemble_class)
            else:
                feature_gen_class = self.get_generator_class(experiment_dict, generator)
                experiment_dict['feature_generator'].update(feature_gen_class)

        return experiment_dict

    def get_generator_class(self, experiment_dict: dict, gen_name: str) -> dict:
        """Combines the name of the generator with the parameters from the config file.

        Args:
            experiment_dict: dictionary with the parameters of the experiment.
            gen_name: name of the generator.

        Returns:
            Dictionary with the name of the generator and its class.

        """
        feature_gen_model = self.feature_generator[gen_name]
        feature_gen_params = experiment_dict['feature_generator_params'].get(gen_name, dict())
        feature_gen_class = {gen_name: feature_gen_model(**feature_gen_params, use_cache=self.use_cache)}
        return feature_gen_class


class DataReader:
    def __init__(self,
                 logger: Logger = None):

        self.logger = logger

    def read(self, dataset_name: str):

        self.logger.info(f'START WORKING on {dataset_name} dataset')
        meta_dict = {}
        # load data
        train_data, test_data = DataLoader(dataset_name).load_data()
        self.logger.info(f'Loaded data from {dataset_name} dataset')
        if train_data is None:
            self.logger.error(f'Some problem with {dataset_name} data. Skip it')
            return None, None, meta_dict
        else:
            n_classes = len(np.unique(train_data[1]))
            meta_dict['Number_of_classes'] = n_classes
            self.logger.info(f'{n_classes} classes detected')
            return train_data, test_data, meta_dict
