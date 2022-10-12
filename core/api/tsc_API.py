import copy
import os
from typing import Any, Union

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
        self.logger = Logger().get_logger()

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
        experiment_dict = copy.deepcopy(self.config_dict)

        experiment_dict['feature_generator'].clear()
        experiment_dict['feature_generator'] = dict()

        for idx, feature_generator in enumerate(self.config_dict['feature_generator']):
            if feature_generator.startswith('ensemble'):
                models = feature_generator.split(': ')[1].split(' ')
                for model in models:
                    feature_generator_class = {
                        model: self.feature_generator_dict[model](**experiment_dict['feature_generator_params'][model])}

                    experiment_dict['feature_generator_params']['ensemble']['list_of_generators'].update(
                        feature_generator_class)

                ensemble_generator = {'ensemble': self.feature_generator_dict['ensemble']
                (**experiment_dict['feature_generator_params']['ensemble'])}

                experiment_dict['feature_generator'].update(ensemble_generator)

            else:
                feature_generator_class = {feature_generator: self.feature_generator_dict[feature_generator]
                (**experiment_dict['feature_generator_params'][feature_generator])}

                experiment_dict['feature_generator'].update(feature_generator_class)

        return experiment_dict

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
        Read yaml config from './experiments/configs/config_name' directory as dictionary file
        :param config_name: yaml-config name
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
            self.config_dict['logger'] = self.logger
            self.logger.info(
                f"Experiment setup:"
                f"\ndatasets - {self.config_dict['datasets_list']},"
                f"\nfeature generators - {self.config_dict['feature_generator']}")

    def run_experiment(self, config_name):
        """
        Run experiment with corresponding config_name
        :param config_name: configuration file name [Config_Classification.yaml]
        """
        self.logger.info(f'START EXPERIMENT')
        experiment_dict = self._init_experiment_setup(config_name)
        launches = self.config_dict['launches']

        classificator = TimeSeriesClassifier(feature_generator_dict=experiment_dict['feature_generator'],
                                             model_hyperparams=experiment_dict['fedot_params'],
                                             ecm_model_flag=experiment_dict['error_correction'])

        for dataset_name in self.config_dict['datasets_list']:
            self.logger.info(f'START WORKING on {dataset_name} dataset')
            try: # to load data
                train_data, test_data = DataLoader(dataset_name).load_data()
                self.logger.info(f'Loaded data from {dataset_name} dataset')
                self._check_window_sizes(dataset_name, train_data)
            except Exception:
                self.logger.error(f'Some problem with {dataset_name} data. Skip it')
                continue

            n_classes = len(np.unique(train_data[1]))
            self.logger.info(f'{n_classes} classes detected')

            classificator.model_hyperparams['metric'] = self._check_metric(n_classes)

            for launch in range(1, launches + 1):
                self.logger.info(f'START LAUNCH {launch}')
                paths_to_save = list(map(lambda x: os.path.join(path_to_save_results(), x, dataset_name, str(launch)),
                                         list(experiment_dict['feature_generator'].keys())))

                self.logger.info('START TRAINING')
                fitted_results = list(map(lambda x: classificator.fit(x, dataset_name), [train_data]))
                fitted_predictor = fitted_results[0]['predictors']
                train_features = fitted_results[0]['train_features']

                self.logger.info('START PREDICTION')
                predictions = classificator.predict(fitted_predictor, test_data, dataset_name)
                predict_on_train = classificator.predict_on_train()

                # n_ecm_cycles = experiment_dict['n_ecm_cycles']
                ecm_fedot_params = dict(problem='regression',
                                        seed=14,
                                        timeout=1,
                                        max_depth=10,
                                        max_arity=4,
                                        cv_folds=2,
                                        logging_level=20,
                                        n_jobs=4)

                ecm_params = dict(n_classes=n_classes,
                                  dataset_name=dataset_name,
                                  save_models=False,
                                  fedot_params=ecm_fedot_params)
                ecm_results = [[]] * len(paths_to_save)

                if self.config_dict['error_correction']:
                    try:
                        self.logger.info('START COMPOSE ECM')
                        ecm_results = list(map(lambda x, y, z, m: ErrorCorrectionModel(**ecm_params,
                                                                                       results_on_test=x,
                                                                                       results_on_train=y,
                                                                                       train_data=(z, train_data[1]),
                                                                                       test_data=(m, test_data[1])).run(),
                                               predictions,
                                               predict_on_train,
                                               train_features,
                                               predictions))
                    except Exception:
                        self.logger.info('ECM COMPOSE WAS FAILED')

                self.logger.info('SAVING RESULTS')

                for i in range(len(paths_to_save)):
                    self.save_results(train_target=train_data[1],
                                      test_target=test_data[1],
                                      path_to_save=paths_to_save[i],
                                      train_features=train_features[i],
                                      prediction=predictions[i],
                                      fitted_predictor=fitted_predictor,
                                      ecm_results=ecm_results[i]),

                # spectral_generators = [x for x in paths_to_save if 'spectral' in x]
                # if len(spectral_generators) != 0:
                #     self._save_spectrum(classificator, path_to_save=spectral_generators)

        self.logger.info('END EXPERIMENT')

    def save_results(self, train_target: Union[np.ndarray, pd.Series],
                     test_target: Union[np.ndarray, pd.Series],
                     path_to_save: str,
                     prediction: Any,
                     train_features: Union[np.ndarray, pd.DataFrame],
                     fitted_predictor,
                     ecm_results: dict):

        metrics, predictions = prediction['metrics'], prediction['prediction']
        predictions_proba, test_features = prediction['predictions_proba'], prediction['test_features']

        path_results = os.path.join(path_to_save, 'test_results')
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        try:
            for predictor in fitted_predictor:
                predictor.current_pipeline.save(path_results)
        except Exception as ex:
            self.logger.error(f'Can not save pipeline: {ex}')

        if len(ecm_results) != 0:
            self.save_boosting_results(**ecm_results, path_to_save=path_results)

        features_names = ['train_features.csv', 'train_target.csv', 'test_features.csv', 'test_target.csv']
        features_list = [train_features, train_target, test_features, test_target]
        _ = list(map(lambda x, y: pd.DataFrame(x).to_csv(os.path.join(path_to_save, y)), features_list, features_names))

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
