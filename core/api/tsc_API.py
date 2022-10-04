import copy
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import yaml

from core.models.ecm.ErrorRunner import ErrorCorrectionModel
from core.models.EnsembleRunner import EnsembleRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.load_data import DataLoader
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
            'ensemble': EnsembleRunner}

    def _init_experiment_setup(self, config_name):
        self.read_yaml_config(config_name)
        experiment_dict = copy.deepcopy(self.config_dict)
        self.use_cache = experiment_dict['use_cache']

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
        try:
            feature_gen_params = experiment_dict['feature_generator_params'][gen_name]
        except KeyError:
            feature_gen_params = dict()
        feature_gen_class = {gen_name: feature_gen_model(**feature_gen_params, use_cache=self.use_cache)}
        return feature_gen_class

    def read_yaml_config(self, config_name: str) -> None:
        """
        Read yaml config from './cases/config/config_name' directory as dictionary file
        :param config_name: yaml-config name, e.g. 'Config_Classification.yaml'
        """
        path = os.path.join(PROJECT_PATH, 'cases', 'config', config_name)
        with open(path, "r") as input_stream:
            self.config_dict = yaml.safe_load(input_stream)
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

        classificator = TimeSeriesClassifier(feature_generator_dict=experiment_dict['feature_generator'],
                                             model_hyperparams=experiment_dict['fedot_params'],
                                             ecm_model_flag=experiment_dict['error_correction'])
        launches = self.config_dict['launches']

        for dataset_name in self.config_dict['datasets_list']:
            self.logger.info(f'START WORKING on {dataset_name} dataset')
            try:  # to load data
                train_data, test_data = DataLoader(dataset_name).load_data()
                self.logger.info(f'Loaded data from {dataset_name} dataset')
            except Exception:
                self.logger.error(f'Some problem with {dataset_name} data. Skip it')
                continue

            n_classes = len(np.unique(train_data[1]))
            self.logger.info(f'{n_classes} classes detected')

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

                if self.config_dict['error_correction']:
                    n_ecm_cycles = experiment_dict['n_ecm_cycles']
                    ecm_fedot_params = dict(problem='regression',
                                            seed=14,
                                            timeout=1,
                                            composer_params=dict(max_depth=10,
                                                                 max_arity=4,
                                                                 cv_folds=2,
                                                                 stopping_after_n_generation=20),
                                            verbose_level=1,
                                            n_jobs=2)

                    ecm_params = dict(n_classes=n_classes,
                                      dataset_name=dataset_name,
                                      save_models=False,
                                      fedot_params=ecm_fedot_params,
                                      n_cycles=n_ecm_cycles)
                    ecm_results = list(map(lambda x, y, z, m: ErrorCorrectionModel(**ecm_params,
                                                                                   results_on_test=x,
                                                                                   results_on_train=y,
                                                                                   train_data=(z, train_data[1]),
                                                                                   test_data=(m, test_data[1])).run(),
                                           predictions,
                                           predict_on_train,
                                           train_features,
                                           predictions))
                else:
                    ecm_results = [None]*len(predictions)

                self.logger.info('SAVING RESULTS')

                _ = list(map(lambda path, x_tr, pred, ecm_res: self.save_results(train_target=train_data[1],
                                                                                 test_target=test_data[1],
                                                                                 path_to_save=path,
                                                                                 train_features=x_tr,
                                                                                 prediction=pred,
                                                                                 fitted_predictor=fitted_predictor,
                                                                                 ecm_results=ecm_res),
                             paths_to_save, train_features, predictions, ecm_results
                             )
                         )

                spectral_generators = [x for x in paths_to_save if 'spectral' in x]
                if len(spectral_generators) != 0:
                    self._save_spectrum(classificator, path_to_save=spectral_generators)

        self.logger.info('END OF EXPERIMENT')

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
        os.makedirs(path_results, exist_ok=True)

        try:
            for predictor in fitted_predictor:
                predictor.current_pipeline.save(path_results)
        except Exception as ex:
            self.logger.error(f'Can not save pipeline: {ex}')

        if ecm_results is not None:
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
    def _get_ts_data(name_of_datasets):
        all_data = list(map(lambda x: DataLoader().load_data(x), name_of_datasets))
        train_data, test_data = [(x[0][0], x[1][0]) for x in all_data], [(x[0][1], x[1][1]) for x in all_data]
        return train_data, test_data

    @staticmethod
    def _save_spectrum(classificator, path_to_save):
        for method, path in zip(list(classificator.composer.dict.keys()), path_to_save):
            pd.concat(classificator.composer.dict[method].eigenvectors_list_train, axis=1).to_csv(
                os.path.join(path, 'train_spectrum.csv'))
            pd.concat(classificator.composer.dict[method].eigenvectors_list_test, axis=1).to_csv(
                os.path.join(path, 'test_spectrum.csv'))
