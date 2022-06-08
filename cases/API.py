import copy
import os
import yaml
import logging
from typing import Dict
from cases.run.utils import read_tsv
from cases.run.QuantileRunner import StatsRunner
from cases.run.SSARunner import SSARunner
from cases.run.SignalRunner import SignalRunner
from cases.run.TopologicalRunner import TopologicalRunner
from core.operation.utils.utils import project_path
from cases.run.ts_clf import TimeSeriesClf
from cases.run.utils import *


class Industrial:
    """ Class-support for performing examples for tasks (read yaml configs, create data folders and log files)"""

    def __init__(self):
        logger = logging.getLogger('Experiment logger')
        logger.setLevel(logging.INFO)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        self.logger = logger

        self.feature_generator_dict = {
            'quantile': StatsRunner,
            'window_quantile': StatsRunner,
            'wavelet': SignalRunner,
            'spectral': SSARunner,
            'spectral_window': SSARunner,
            'topological': TopologicalRunner}

    def _get_ts_data(self, name_of_datasets):
        all_data = list(map(lambda x: read_tsv(x), name_of_datasets))
        train_data, test_data = [(x[0][0], x[1][0]) for x in all_data], [(x[0][1], x[1][1]) for x in all_data]
        return train_data, test_data

    def read_yaml_config(self, config_name: str) -> Dict:
        """ Read yaml config from './experiments/configs/config_name' directory as dictionary file
            :param config_name: yaml-config name
            :return: yaml config
        """
        path = os.path.join(project_path(), 'cases', 'config', config_name)
        with open(path, "r") as input_stream:
            self.config_dict = yaml.safe_load(input_stream)
            self.config_dict['logger'] = self.logger
            self.logger.info(f"schema ready: {self.config_dict}")

    def fit(self):
        pass

    def run_experiment(self, config_name):

        self.read_yaml_config(config_name)
        experiment_dict = copy.deepcopy(self.config_dict)

        experiment_dict['feature_generator'].clear()
        experiment_dict['feature_generator'] = dict()

        for idx, feature_generator in enumerate(self.config_dict['feature_generator']):
            experiment_dict['feature_generator'].update(
                {feature_generator: self.feature_generator_dict[feature_generator]
                (fedot_params=experiment_dict['fedot_params'],
                 **experiment_dict['feature_generator_params'][feature_generator])})

        classificator = TimeSeriesClf(feature_generator_dict=experiment_dict['feature_generator'],
                                      model_hyperparams=experiment_dict['fedot_params'])

        train_archive, test_archive = self._get_ts_data(self.config_dict['datasets_list'])

        for train_data, test_data in zip(train_archive, test_archive):
            fitted_predictor = list(map(lambda x: classificator.fit(x), [train_data]))
            prediction = list(map(lambda x: classificator.predict(fitted_predictor, x), [test_data]))
            # self.path_to_save = self._create_path_to_save(method, dataset, launch)
            _ = 1
