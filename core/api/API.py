from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split

from core.api.utils.checkers_collections import *
from core.api.utils.method_collections import *
from core.api.utils.reader_collections import *
from core.api.utils.saver_collections import ResultSaver
from core.architecture.abstraction.LoggerSingleton import Logger
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.utils.utils import path_to_save_results


class Industrial(Fedot):
    """Class-support for performing experiments for tasks (read yaml configs, create data folders and log files)

    Attributes:
        config_dict (dict): dictionary with the parameters of the experiment
        logger (Logger): logger instance for logging
        feature_generator_dict (dict): dictionary with the names of the generators and their classes
    """

    def __init__(self):
        super(Fedot, self).__init__()
        self.config_dict = None
        self.logger = Logger.__call__().get_logger()
        self.fitted_model = None
        self.labels, self.prediction_proba = None, None
        self.test_features, self.train_features = None, None
        self.feature_generator_dict = {method.name: method.value for method in FeatureGenerator}
        self.feature_generator_dict.update({method.name: method.value for method in WindowFeatureGenerator})
        self.ensemble_methods_dict = {method.name: method.value for method in EnsembleGenerator}
        self.task_pipeline_dict = {method.name: method.value for method in TaskGenerator}

        self.YAML = YamlReader(feature_generator=self.feature_generator_dict)
        self.reader = DataReader()
        self.checker = ParameterCheck()
        self.saver = ResultSaver()

    def fit(self, **kwargs):
        self._define_data()
        self.fitted_model, self.train_features = self._obtain_model(**kwargs)
        return self.fitted_model, self.train_features

    def predict(self, features: tuple, save_predictions: bool = False) -> Union[dict, tuple]:
        if self.model_composer is None:
            raise ('Fedot model was not fitted'.center(50, '-'))
        else:
            prediction = self.model_composer.predict(test_features=features)
            self.labels, self.test_features = prediction['label'], prediction['test_features']

            if len(self.labels.shape) > 1:
                self.labels = np.argmax(self.labels, axis=1)
                prediction['labels'] = self.labels

            if save_predictions:
                return prediction
            else:
                return self.labels, self.test_features

    def predict_proba(self,
                      features: tuple,
                      save_predictions: bool = False,
                      probs_for_all_classes: bool = False):
        prediction = self.model_composer.predict_proba(features)
        if save_predictions:
            return prediction
        else:
            self.prediction_proba, self.test_features = prediction['class_probability'], prediction['test_features']
            return self.prediction_proba, self.test_features

    def get_metrics(self,
                    target: np.ndarray,
                    prediction_label: np.ndarray,
                    prediction_proba: np.ndarray) -> dict:
        metrics_dict = PerformanceAnalyzer().calculate_metrics(target=target,
                                                               predicted_labels=prediction_label,
                                                               predicted_probs=prediction_proba)

        return metrics_dict

    def _define_data(self):
        pass

    def _obtain_model(self,
                      model_name: str,
                      task_type: str,
                      train_features: pd.DataFrame,
                      train_target: np.ndarray,
                      feature_generator_params: dict = None,
                      model_params: dict = None,
                      dataset_name: str = None,
                      ecm_mode: bool = False):
        try:
            generator_params = self.config_dict['feature_generator_params'][model_name]
        except KeyError:
            generator_params = feature_generator_params

        generator = self.feature_generator_dict[model_name](**generator_params)

        self.model_composer = self.task_pipeline_dict[task_type](generator_name=model_name,
                                                                 generator_runner=generator,
                                                                 model_hyperparams=model_params,
                                                                 ecm_model_flag=ecm_mode)

        metric = self.checker.check_metric_type(train_target)
        baseline_type = self.checker.check_baseline_type(self.config_dict, model_params)
        self.model_composer.model_hyperparams['metric'] = metric
        self.logger.info(f'Fitting model...')

        fitted_model, train_features = self.model_composer.fit(train_features=train_features,
                                                               train_target=train_target,
                                                               dataset_name=dataset_name,
                                                               baseline_type=baseline_type)
        return fitted_model, train_features

    def run_experiment(self, config: Union[str, dict], direct_path: bool = False):
        """Run experiment with corresponding config_name.

        Args:
            config: path to config file or dictionary with parameters.
            direct_path: if True, then config_path is an absolute path to the config file. Otherwise, Industrial will
            search for the config file in the config folders.

        """
        self.logger.info(f'START EXPERIMENT'.center(50, '-'))
        experiment_results = {}

        self.config_dict = self.YAML.init_experiment_setup(config,
                                                           direct_path=direct_path)

        for dataset_name in self.config_dict['datasets_list']:
            experiment_results[dataset_name] = {}
            experiment_dict = copy.deepcopy(self.config_dict)
            n_cycles = experiment_dict['launches']

            result = self._run_modelling_cycle(experiment_dict=experiment_dict,
                                               task_type=self.config_dict['task'],
                                               n_cycles=n_cycles,
                                               dataset_name=dataset_name)
            experiment_results[dataset_name]['Original'] = result

            if self.config_dict['error_correction']:
                experiment_results[dataset_name]['ECM'] = self.apply_ECM(
                    modelling_results=experiment_results[dataset_name]['Original'])

            if self.config_dict['ensemble_algorithm']:
                ensemble_result = self.apply_ensemble(modelling_results=result,
                                                      ensemble_mode=self.config_dict['ensemble_algorithm'])

                experiment_results[dataset_name]['Ensemble'] = ensemble_result

            self.save_results(modelling_results=experiment_results,
                              dataset_name=dataset_name)

        self.logger.info('END OF EXPERIMENT'.center(50, '-'))

    def _run_modelling_cycle(self,
                             experiment_dict: dict,
                             task_type: str,
                             n_cycles: int,
                             dataset_name: str,
                             ):
        self.logger.info(f'TYPE OF ML TASK - {task_type}'.center(50, '-'))
        modelling_results = dict()
        train_data, test_data, n_classes = self.reader.read(dataset_name=dataset_name)

        if train_data is None:
            return None

        experiment_dict = self.exclude_generators(experiment_dict, train_data[0])

        for runner_name, runner in experiment_dict['feature_generator'].items():
            modelling_results[runner_name] = {}
            for launch in range(1, n_cycles + 1):
                try:
                    runner_result = {}
                    paths_to_save = os.path.join(path_to_save_results(), runner_name, dataset_name, str(launch))

                    fitted_predictor, train_features = self._obtain_model(model_name=runner_name,
                                                                          task_type=task_type,
                                                                          train_features=train_data[0],
                                                                          dataset_name=dataset_name,
                                                                          train_target=train_data[1],
                                                                          model_params=experiment_dict['fedot_params'],
                                                                          ecm_mode=experiment_dict['error_correction'])

                    runner_result['fitted_predictor'], runner_result[
                        'train_features'] = fitted_predictor, train_features
                    runner_result['test_target'] = test_data[1]
                    runner_result['path_to_save'] = paths_to_save
                    runner_result['train_target'] = train_data[1]

                    self.logger.info(f'START PREDICTION AT LAUNCH-{launch}'.center(50, '-'))

                    label_dict = self.predict(features=test_data[0],
                                              save_predictions=True)
                    proba_dict = self.predict_proba(features=test_data[0],
                                                    save_predictions=True)
                    runner_result.update(label_dict)
                    runner_result['class_probability'] = proba_dict['class_probability']
                    runner_result['metrics'] = self.get_metrics(target=runner_result['test_target'],
                                                                prediction_proba=runner_result['class_probability'],
                                                                prediction_label=runner_result['label'])
                    self.logger.info(f'METRICS AT TEST DATASET'.center(50, '-'))
                    metric_df = runner_result['metrics']

                    self.logger.info(f'{metric_df}')

                    modelling_results[runner_name][launch] = runner_result

                except Exception as ex:
                    self.logger.info(f'PROBLEM WITH {runner_name} AT LAUNCH {launch}. REASON - {ex}')

        return modelling_results

    def exclude_generators(self, experiment_dict: dict, train_data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Method that excludes generators that can't be applied to the dataset due to the length of time series.

        Args:
            experiment_dict: dictionary with experiment parameters.
            train_data: dataframe with train features.

        Returns:
            Same experiment_dict with excluded generators.

        """
        exclusion_list = ['topological', 'spectral', 'window_spectral']
        ts_length = train_data.shape[1]
        if ts_length > 800:

            for exclude in exclusion_list:
                if exclude in experiment_dict['feature_generator']:
                    experiment_dict['feature_generator'].remove(exclude)
                    experiment_dict['feature_generator_params'].pop(exclude, None)
            self.logger.info(f'Time series length is too long ({ts_length}>800), exclude {exclusion_list} generators')

        return experiment_dict

    def save_results(self, dataset_name, modelling_results: dict):
        result_at_dataset = modelling_results[dataset_name]
        for approach in result_at_dataset:
            if result_at_dataset[approach] is not None:
                for model in result_at_dataset[approach]:
                    self.saver.save_method_dict[approach](prediction=result_at_dataset[approach][model])

    def apply_ECM(self, modelling_results: Union[dict, list]):
        self.logger.info('START FIT ERROR CORRECTION MODEL'.center(50, '-'))
        predict_on_train = modelling_results['model'].predict_on_train()
        ecm_fedot_params = modelling_results['experiment_dict']['model_params']
        ecm_fedot_params['problem'] = 'regression'
        ecm_fedot_params['timeout'] = round(modelling_results['experiment_dict']['model_params']['timeout'] / 3)

        # ecm_params = dict(n_classes=n_classes,
        #                   dataset_name=dataset_name,
        #                   save_models=False,
        #                   fedot_params=ecm_fedot_params)
        #
        # try:
        #     self.logger.info('START COMPOSE ECM')
        #     ecm_results = ErrorCorrectionModel(**ecm_params,
        #                                        results_on_test=predictions,
        #                                        results_on_train=predict_on_train,
        #                                        train_data=train_data,
        #                                        test_data=test_data,
        #                                        ).run()
        # except Exception:
        #     self.logger.info('ECM COMPOSE WAS FAILED')
        return predict_on_train

    def apply_ensemble(self,
                       modelling_results: Union[dict, list],
                       ensemble_mode: str = 'AGG_voting'):
        single_mode = True
        if self.config_dict is not None:
            ensemble_mode = self.config_dict['ensemble_algorithm']
            single_mode = False
        ensemble_model = self.ensemble_methods_dict[ensemble_mode]()
        ensemble_results = ensemble_model.ensemble(modelling_results=modelling_results,
                                                   single_mode=single_mode)
        return ensemble_results

    @staticmethod
    def _create_validation_dataset(train_data):
        X_train, X_val, y_train, y_val = train_test_split(train_data[0],
                                                          train_data[1],
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=train_data[1])
        train_data, validation_data = (X_train, y_train), (X_val, y_val)
        return train_data, validation_data
