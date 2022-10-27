import copy
import os
from typing import Union

import numpy as np
import pandas as pd
import yaml

from core.models.ecm.ErrorRunner import ErrorCorrectionModel
from core.models.EnsembleRunner import EnsembleRunner
from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner
from core.operation.utils.load_data import DataLoader
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import path_to_save_results
from core.operation.utils.utils import PROJECT_PATH
from core.TimeSeriesClassifier import TimeSeriesClassifier


class Industrial:
    """
    Class-support for performing experiments for tasks (read yaml configs, create data folders and log files)
    """

    def __init__(self):
        self.config_dict = None
        self.logger = Logger.__call__().get_logger()

        self.feature_generator_dict = {
            'quantile': StatsRunner,
            'window_quantile': StatsRunner,
            'wavelet': SignalRunner,
            'spectral': SSARunner,
            'spectral_window': SSARunner,
            'topological': TopologicalRunner,
            'recurrence': RecurrenceRunner,
            'ensemble': EnsembleRunner}

    def _init_experiment_setup(self, config_name):
        self.read_yaml_config(config_name=config_name)
        for dataset_name in self.config_dict['datasets_list']:
            train_data, _ = DataLoader(dataset_name).load_data()
            self._check_window_sizes(dataset_name=dataset_name, train_data=train_data)
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

    def get_generator_class(self, experiment_dict, gen_name):
        """
        Combines the name of the generator with the parameters from the config file
        :return: dictionary with the name of the generator and its parameters
        """
        feature_gen_model = self.feature_generator_dict[gen_name]
        feature_gen_params = experiment_dict['feature_generator_params'].get(gen_name, dict())
        feature_gen_class = {gen_name: feature_gen_model(**feature_gen_params, use_cache=self.use_cache)}
        return feature_gen_class

    def _check_window_sizes(self, dataset_name, train_data):
        for key in self.config_dict['feature_generator_params'].keys():
            if key.startswith('spectral'):
                self.logger.info(f'CHECK WINDOW SIZES FOR DATASET-{dataset_name} AND {key} method')
                if dataset_name not in self.config_dict['feature_generator_params'][key].keys():
                    ts_length = train_data[0].shape[1]
                    list_of_WS = list(map(lambda x: round(ts_length / x), [10, 5, 3]))
                    self.config_dict['feature_generator_params'][key]['window_sizes'][dataset_name] = list_of_WS
                    self.logger.info(f'THERE ARE NO PREDEFINED WINDOWS. '
                                     f'DEFAULTS WINDOWS SIZES WAS SET - {list_of_WS}. '
                                     f'THATS EQUAL 10/20/30% OF TS LENGTH')

    def _check_metric(self, n_classes):
        if n_classes > 2:
            self.logger.info('Metric for optimization - F1')
            return 'f1'
        else:
            self.logger.info('Metric for optimization - ROC_AUC')
            return 'roc_auc'

    def read_yaml_config(self, config_name: str) -> None:
        """
        Read yaml config from './cases/config/config_name' directory as dictionary file
        :param config_name: yaml-config name, e.g. 'Config_Classification.yaml'
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

    def run_experiment(self, config_name):
        """
        Run experiment with corresponding config_name
        :param config_name: configuration file name [Config_Classification.yaml]
        """
        self.logger.info(f'START EXPERIMENT')
        experiment_dict = self._init_experiment_setup(config_name)
        launches = self.config_dict['launches']

        for dataset_name in self.config_dict['datasets_list']:
            self.logger.info(f'START WORKING on {dataset_name} dataset')

            # load data
            train_data, test_data = DataLoader(dataset_name).load_data()
            self.logger.info(f'Loaded data from {dataset_name} dataset')
            if train_data is None:
                self.logger.error(f'Some problem with {dataset_name} data. Skip it')
                continue

            n_classes = len(np.unique(train_data[1]))
            self.logger.info(f'{n_classes} classes detected')

            for runner_name, runner in experiment_dict['feature_generator'].items():
                self.logger.info(f'{runner_name} runner is on duty')

                classificator = TimeSeriesClassifier(generator_name=runner_name,
                                                     generator_runner=runner,
                                                     model_hyperparams=experiment_dict['fedot_params'],
                                                     ecm_model_flag=experiment_dict['error_correction'])
                classificator.feature_generator_params = self.config_dict['feature_generator_params']
                classificator.model_hyperparams['metric'] = self._check_metric(n_classes)

                for launch in range(1, launches + 1):
                    self.logger.info(f'START LAUNCH {launch}')
                    self.logger.info('START TRAINING')
                    fitted_predictor, train_features = classificator.fit(train_data, dataset_name)

                    self.logger.info('START PREDICTION')
                    predictions = classificator.predict(test_data, dataset_name)

                    if self.config_dict['error_correction']:
                        predict_on_train = classificator.predict_on_train()
                        ecm_fedot_params = experiment_dict['fedot_params']
                        ecm_fedot_params['problem'] = 'regression'
                        ecm_fedot_params['timeout'] = round(experiment_dict['fedot_params']['timeout'] / 3)

                        ecm_params = dict(n_classes=n_classes,
                                          dataset_name=dataset_name,
                                          save_models=False,
                                          fedot_params=ecm_fedot_params)

                        try:
                            self.logger.info('START COMPOSE ECM')
                            ecm_results = ErrorCorrectionModel(**ecm_params,
                                                               results_on_test=predictions,
                                                               results_on_train=predict_on_train,
                                                               train_data=train_data,
                                                               test_data=test_data,
                                                               ).run()
                        except Exception:
                            self.logger.info('ECM COMPOSE WAS FAILED')
                    else:
                        ecm_results = None

                    self.logger.info('*------------SAVING RESULTS------------*')
                    paths_to_save = os.path.join(path_to_save_results(), runner_name, dataset_name, str(launch))

                    self.save_results(train_target=train_data[1],
                                      test_target=test_data[1],
                                      path_to_save=paths_to_save,
                                      train_features=train_features,
                                      prediction=predictions,
                                      fitted_predictor=fitted_predictor,
                                      ecm_results=ecm_results),

                    # spectral_generators = [x for x in paths_to_save if 'spectral' in x]
                    # if len(spectral_generators) != 0:
                    #     self._save_spectrum(classificator, path_to_save=spectral_generators)

        self.logger.info('END OF EXPERIMENT')

    def save_results(self, train_target: Union[np.ndarray, pd.Series],
                     test_target: Union[np.ndarray, pd.Series],
                     path_to_save: str,
                     prediction: dict,
                     train_features: Union[np.ndarray, pd.DataFrame],
                     fitted_predictor,
                     ecm_results: dict):

        metrics, predictions = prediction['metrics'], prediction['prediction']
        predictions_proba, test_features = prediction['predictions_proba'], prediction['test_features']

        path_results = os.path.join(path_to_save, 'test_results')
        os.makedirs(path_results, exist_ok=True)

        try:
            fitted_predictor.current_pipeline.save(path_results)
        except Exception as ex:
            self.logger.error(f'Can not save pipeline: {ex}')

        if ecm_results:
            self.save_boosting_results(**ecm_results, path_to_save=path_results)

        features_names = ['train_features.csv', 'train_target.csv', 'test_features.csv', 'test_target.csv']
        features_list = [train_features, train_target, test_features, test_target]

        for name, features in zip(features_names, features_list):
            pd.DataFrame(features).to_csv(os.path.join(path_to_save, name))


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

        for p, d in zip(['probs_preds_target.csv', 'metrics.csv'],
                        [df_preds, df_metrics]):
            full_path = os.path.join(path_results, p)
            d.to_csv(full_path)

    def save_boosting_results(self, solution_table, metrics_table, model_list, ensemble_model, path_to_save):
        location = os.path.join(path_to_save, 'boosting')
        if not os.path.exists(location):
            os.makedirs(location)
        solution_table.to_csv(os.path.join(location, 'solution_table.csv'))
        metrics_table.to_csv(os.path.join(location, 'metrics_table.csv'))

        models_path = os.path.join(location, 'boosting_pipelines')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for index, model in enumerate(model_list):
            try:
                model.current_pipeline.save(path=os.path.join(models_path, f'boost_{index}'),
                                            datetime_in_path=False)
            except Exception:
                model.save(path=os.path.join(models_path, f'boost_{index}'),
                           datetime_in_path=False)
        if ensemble_model is not None:
            try:
                ensemble_model.current_pipeline.save(path=os.path.join(models_path, 'boost_ensemble'),
                                                     datetime_in_path=False)
            except Exception:
                ensemble_model.save(path=os.path.join(models_path, 'boost_ensemble'),
                                    datetime_in_path=False)
        else:
            self.logger.info('Ensemble model cannot be saved due to applied SUM method ')

    @staticmethod
    def _save_spectrum(classificator, path_to_save):
        for method, path in zip(list(classificator.composer.dict.keys()), path_to_save):
            if 'spectral' in method:
                pd.concat(classificator.composer.dict[method].eigenvectors_list_train, axis=1).to_csv(
                    os.path.join(path, 'train_spectrum.csv'))
                pd.concat(classificator.composer.dict[method].eigenvectors_list_test, axis=1).to_csv(
                    os.path.join(path, 'test_spectrum.csv'))
