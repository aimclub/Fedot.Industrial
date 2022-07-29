import copy
from typing import Union

import numpy as np
import yaml

from cases.run.EnsembleRunner import EnsembleRunner
from cases.run.QuantileRunner import StatsRunner
from cases.run.SSARunner import SSARunner
from cases.run.SignalRunner import SignalRunner
from cases.run.TimeSeriesClassifier import TimeSeriesClassifier
from cases.run.TopologicalRunner import TopologicalRunner
from cases.run.utils import *
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import path_to_save_results


class Industrial:
    """ Class-support for performing examples for tasks
    (read yaml configs, create data folders and log files)"""

    def __init__(self):
        self.logger = Logger().get_logger()

        self.feature_generator_dict = {
            'quantile': StatsRunner,
            'window_quantile': StatsRunner,
            'wavelet': SignalRunner,
            'spectral': SSARunner,
            'spectral_window': SSARunner,
            'topological': TopologicalRunner,
            'ensemble': EnsembleRunner}

    def _get_ts_data(self, name_of_datasets):
        all_data = list(map(lambda x: read_tsv(x), name_of_datasets))
        train_data, test_data = [(x[0][0], x[1][0]) for x in all_data], [(x[0][1], x[1][1]) for x in all_data]
        return train_data, test_data

    def _init_experiment_setup(self, config_name):
        self.read_yaml_config(config_name)
        experiment_dict = copy.deepcopy(self.config_dict)

        experiment_dict['feature_generator'].clear()
        experiment_dict['feature_generator'] = dict()

        for idx, feature_generator in enumerate(self.config_dict['feature_generator']):
            if feature_generator.startswith('ensemble'):
                models = feature_generator.split(': ')[1].split(' ')
                for model in models:
                    feature_generator_class = {model: self.feature_generator_dict[model]
                    (fedot_params=experiment_dict['fedot_params'],
                     **experiment_dict['feature_generator_params'][model])}

                    experiment_dict['feature_generator_params']['ensemble']['list_of_generators'].update(
                        feature_generator_class)

                ensemble_generator = {'ensemble': self.feature_generator_dict['ensemble']
                (fedot_params=experiment_dict['fedot_params'],
                 **experiment_dict['feature_generator_params']['ensemble'])}

                experiment_dict['feature_generator'].update(ensemble_generator)

            else:
                feature_generator_class = {feature_generator: self.feature_generator_dict[feature_generator]
                (fedot_params=experiment_dict['fedot_params'],
                 **experiment_dict['feature_generator_params'][feature_generator])}

                experiment_dict['feature_generator'].update(feature_generator_class)

        return experiment_dict

    def _save_spectrum(self, classificator, path_to_save):
        for method, path in zip(list(classificator.composer.dict.keys()), path_to_save):
            pd.concat(classificator.composer.dict[method].eigenvectors_list_train, axis=1).to_csv(
                os.path.join(path, 'train_spectrum.csv'))
            pd.concat(classificator.composer.dict[method].eigenvectors_list_test, axis=1).to_csv(
                os.path.join(path, 'test_spectrum.csv'))

    def read_yaml_config(self, config_name: str) -> None:
        """ Read yaml config from './experiments/configs/config_name' directory as dictionary file
            :param config_name: yaml-config name
        """
        path = os.path.join(project_path(), 'cases', 'config', config_name)
        with open(path, "r") as input_stream:
            self.config_dict = yaml.safe_load(input_stream)
            self.config_dict['logger'] = self.logger
            self.logger.info(
                f"Schema ready:\ndatasets - {self.config_dict['datasets_list']},\nfeature generators - {self.config_dict['feature_generator']}")

    @staticmethod
    def save_results(predictions: Union[np.ndarray, pd.DataFrame],
                     predictions_proba: Union[np.ndarray, pd.DataFrame],
                     train_target: Union[np.ndarray, pd.Series],
                     test_target: Union[np.ndarray, pd.Series],
                     path_to_save: str,
                     metrics: dict,
                     train_features: Union[np.ndarray, pd.DataFrame],
                     test_features: Union[np.ndarray, pd.DataFrame],
                     ):

        path_results = os.path.join(path_to_save, 'test_results')
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        features_names = ['train_features.csv', 'train_target.csv', 'test_features.csv', 'test_target.csv']
        features_list = [train_features, train_target, test_features, test_target]
        saved_features = list(
            map(lambda x, y: pd.DataFrame(x).to_csv(os.path.join(path_to_save, y)), features_list, features_names))

        if type(predictions_proba) is not pd.DataFrame:
            df_preds = pd.DataFrame(predictions_proba)
            df_preds['Target'] = test_target
            df_preds['Preds'] = predictions
        else:
            df_preds = predictions_proba
            df_preds['Target'] = test_target.values

        if type(metrics) is str:
            df_metrics = pd.DataFrame()
        else:
            df_metrics = pd.DataFrame.from_records(data=[x for x in metrics.items()]).reset_index()

        # df_metrics['Inference'] = inference
        # df_metrics['Fit_time'] = fit_time
        # df_metrics['window'] = window

        for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                        [df_preds, df_metrics]):
            full_path = os.path.join(path_results, p)
            d.to_csv(full_path)

    def run_experiment(self, config_name):

        experiment_dict = self._init_experiment_setup(config_name)

        classificator = TimeSeriesClassifier(feature_generator_dict=experiment_dict['feature_generator'],
                                             model_hyperparams=experiment_dict['fedot_params'])

        train_archive, test_archive = self._get_ts_data(self.config_dict['datasets_list'])
        launch = self.config_dict['launches']

        for train_data, test_data, dataset_name in zip(train_archive, test_archive, self.config_dict['datasets_list']):
            paths_to_save = list(map(lambda x: os.path.join(path_to_save_results(), x, dataset_name, str(launch)),
                                     list(experiment_dict['feature_generator'].keys())))

            fitted_results = list(map(lambda x: classificator.fit(x, dataset_name), [train_data]))

            fitted_predictor = fitted_results[0]['predictors']
            train_features = fitted_results[0]['train_features']

            predictions = list(
                map(lambda x: classificator.predict(fitted_predictor, x, dataset_name), [test_data]))

            metrics = predictions[0]['metrics']
            test_features = predictions[0]['test_features']
            prediction = predictions[0]['prediction']
            prediction_proba = predictions[0]['prediction_proba']

            saved_result = list(map(lambda x, y, z, k, j, m: self.save_results(train_target=train_data[1],
                                                                               test_target=test_data[1],
                                                                               path_to_save=x,
                                                                               train_features=y,
                                                                               test_features=z,
                                                                               metrics=k,
                                                                               predictions=j,
                                                                               predictions_proba=m),
                                    paths_to_save, train_features, test_features, metrics, prediction,
                                    prediction_proba))

            spectral_generators = [x for x in paths_to_save if 'spectral' in x]
            if len(spectral_generators) != 0:
                self._save_spectrum(classificator, path_to_save=spectral_generators)
