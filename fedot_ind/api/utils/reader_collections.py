import logging
import os
from typing import Union

import numpy as np
import yaml

from fedot_ind.api.utils.checkers_collections import ParameterCheck
from fedot_ind.api.utils.hp_generator_collection import GeneratorParams
from fedot_ind.core.architecture.settings.pipeline_factory import FeatureGenerator
from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.architecture.utils.utils import PROJECT_PATH
from fedot_ind.core.models.BaseExtractor import BaseExtractor


class YamlReader:
    """
    Class for reading the config file for the experiment.

    Attributes:
        config_dict: dictionary with the parameters of the experiment.
        experiment_check: class for checking the correctness of the parameters.
    """

    def __init__(self):

        self.config_dict = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.experiment_check = ParameterCheck()
        self.experiment_dict = None

    def read_yaml_config(self,
                         # config_path: str,
                         # direct_path: bool = False,
                         # return_dict: bool = False
                         ) -> Union[None, dict]:
        """Read yaml config from directory as dictionary file.

        Args:
            config_path: path to the config file.
            direct_path: flag that indicates whether to use direct path to config file or read it from framework
            directory. config_path: name of the config file.
            return_dict: flag that indicates whether to return the dictionary with the parameters of the experiment.

        Returns:
            None

        """
        # if direct_path:
        #     path = config_path
        # else:
        #     path = os.path.join(PROJECT_PATH, config_path)
        #
        # with open(path, "r") as input_stream:
        #     config_dict = yaml.safe_load(input_stream)
        #     if 'path_to_config' in list(config_dict.keys()):
        #         config_path = config_dict['path_to_config']
        #         path = os.path.join(PROJECT_PATH, config_path)
        #         with open(path, "r") as input_stream:
        #             config_dict_template = yaml.safe_load(input_stream)
        #         config_dict = {**config_dict_template, **config_dict}
        #         del config_dict['path_to_config']
        #
        #     if 'baseline' not in config_dict.keys():
        #         config_dict['baseline'] = None
        #
        # if return_dict:
        #     return config_dict

        return dict(task='ts_classification',
                    dataset=None,
                    strategy='fedot_preset',
                    model_params={'problem': 'classification',
                                  'seed': 42,
                                  'timeout': 15,
                                  'max_depth': 10,
                                  'max_arity': 4,
                                  'cv_folds': 2,
                                  'logging_level': 20,
                                  'n_jobs': -1}
                    )

    def init_experiment_setup(self, **kwargs) -> dict:
        """Initializes the experiment setup with provided config file or dictionary.

        Args:
            kwargs: parameters of the experiment.

        Returns:
            Dictionary with the parameters of the experiment.

        """

        # base_config_path = os.path.join(PROJECT_PATH, 'api', 'config.yaml')
        # self.experiment_dict = self.read_yaml_config(config_path=base_config_path,
        #                                              return_dict=True)
        self.experiment_dict = self.read_yaml_config()

        fedot_config = {}
        industrial_config = {k: v if k not in ['timeout', 'n_jobs', 'logging_level'] else fedot_config.update({k: v})
                             for k, v in kwargs.items()}

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
                generators_to_ensemble = generator.split(': ')[1].split(' ')
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
        feature_gen_params = GeneratorParams[generator].value

        for param in feature_gen_params:
            feature_gen_params[param] = self.experiment_dict.get(param, feature_gen_params[param])

        feature_gen_class = feature_gen_model(feature_gen_params)
        return feature_gen_class

    def __report_experiment_setup(self, experiment_dict):

        self.logger.info(f'''Experiment setup:
        dataset - {experiment_dict['dataset']},
        strategy - {experiment_dict['strategy']},
        use_cache - {experiment_dict['use_cache']},
        n_jobs - {experiment_dict['model_params']['n_jobs']},
        timeout - {experiment_dict['model_params']['timeout']}''')


class DataReader:
    """
    Class for reading train and test data from the dataset. Exploits ``DataLoader`` class to read or download data
    from the UCR time series archive.

    """

    def __init__(self):

        self.logger = logging.getLogger(self.__class__.__name__)

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
