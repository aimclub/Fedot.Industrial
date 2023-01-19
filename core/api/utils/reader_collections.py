import copy
import os
from typing import Union

import numpy as np
import yaml
from fedot.core.log import default_log as Logger

from core.api.utils.checkers_collections import ParameterCheck
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.architecture.utils.utils import PROJECT_PATH


class YamlReader:
    """
    Class for reading the config file for the experiment.

    Args:
        feature_generator: dictionary with the names of the generators and their classes.

    Attributes:
        config_dict: dictionary with the parameters of the experiment.
        experiment_check: class for checking the correctness of the parameters.
        use_cache: flag that indicates whether to use the cache.
    """

    def __init__(self, feature_generator: dict = None):

        self.config_dict = None
        self.feature_generator = feature_generator
        self.logger = Logger(self.__class__.__name__)
        self.experiment_check = ParameterCheck()
        self.use_cache = None

    def read_yaml_config(self,
                         config_path: str,
                         direct_path: bool = False,
                         return_dict: bool = False) -> None:
        """Read yaml config from directory as dictionary file.

        Args:
            config_path: path to the config file.
            direct_path: flag that indicates whether to use direct path to config file or read it from framework
            directory. config_path: name of the config file.
            return_dict: flag that indicates whether to return the dictionary with the parameters of the experiment.

        Returns:
            None

        """
        if direct_path:
            path = config_path
        else:
            path = os.path.join(PROJECT_PATH, config_path)

        with open(path, "r") as input_stream:
            self.config_dict = yaml.safe_load(input_stream)
            if 'path_to_config' in list(self.config_dict.keys()):
                config_path = self.config_dict['path_to_config']
                path = os.path.join(PROJECT_PATH, config_path)
                with open(path, "r") as input_stream:
                    config_dict_template = yaml.safe_load(input_stream)
                self.config_dict = {**config_dict_template, **self.config_dict}
                del self.config_dict['path_to_config']

            if 'baseline' not in self.config_dict.keys():
                self.config_dict['baseline'] = None

        if return_dict:
            return self.config_dict

    def init_experiment_setup(self, config: Union[str, dict],
                              direct_path: bool = False) -> dict:
        """Initializes the experiment setup with provided config file or dictionary.

        Args:
            config: dictionary with the parameters of the experiment OR path to the config file.
            direct_path: flag that indicates whether to use direct path to config file or read it from framework
            directory.

        Returns:
            Dictionary with the parameters of the experiment.

        """

        if isinstance(config, dict):
            base_config_path = 'cases/config/Config_Classification.yaml'
            self.read_yaml_config(config_path=base_config_path)

            industrial_config = {k: v for k, v in config.items() if k not in ['timeout', 'n_jobs']}

            self.config_dict.update(**industrial_config)
            self.config_dict['fedot_params']['timeout'] = config['timeout']
            self.config_dict['fedot_params']['n_jobs'] = config['n_jobs']

        elif isinstance(config, str):
            self.read_yaml_config(config_path=config,
                                  direct_path=direct_path)
        else:
            self.logger.error('Wrong type of config file')
            raise ValueError('Config must be a string or a dictionary!')

        experiment_dict = copy.deepcopy(self.config_dict)
        self.use_cache = experiment_dict['use_cache']

        experiment_dict['feature_generator'].clear()
        experiment_dict['feature_generator'] = dict()

        for idx, generator in enumerate(self.config_dict['feature_generator']):
            if generator.startswith('ensemble'):
                generators = generator.split(': ')[1].split(' ')
                for gen_name in generators:
                    feature_gen_class = self._get_generator_class(experiment_dict, gen_name)
                    experiment_dict['feature_generator_params']['ensemble']['list_of_generators'].update(
                        feature_gen_class)

                ensemble_class = self._get_generator_class(experiment_dict, 'ensemble')
                experiment_dict['feature_generator'].update(ensemble_class)
            else:
                feature_gen_class = self._get_generator_class(experiment_dict, generator)
                experiment_dict['feature_generator'].update(feature_gen_class)

        self.__report_experiment_setup(experiment_dict)

        return experiment_dict

    def _get_generator_class(self, experiment_dict: dict, gen_name: str) -> dict:
        """Support method that combines the name of the generator with the parameters from the config file.

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

    def __report_experiment_setup(self, experiment_dict):

        self.logger.info(f'''Experiment setup:
        datasets - {experiment_dict['datasets_list']},
        feature generators - {list(experiment_dict['feature_generator'])},
        use_cache - {experiment_dict['use_cache']},
        error_correction - {experiment_dict['error_correction']},
        n_jobs - {experiment_dict['fedot_params']['n_jobs']}''')


class DataReader:
    """
    Class for reading train and test data from the dataset. Exploits ``DataLoader`` class to read or download data
    from the UCR time series archive.

    """

    def __init__(self):

        self.logger = Logger(self.__class__.__name__)

    def read(self, dataset_name: str):

        # load data
        train_data, test_data = DataLoader(dataset_name).load_data()
        if train_data is None:
            self.logger.error(f'Some problem with {dataset_name} data. Skip it')
            return None, None, None
        else:
            n_classes = len(np.unique(train_data[1]))
            self.logger.info(f'Loaded data from {dataset_name} local data folder')
            self.logger.info(f'{n_classes} classes detected')
            return train_data, test_data, n_classes
