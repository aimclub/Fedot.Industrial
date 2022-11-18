from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split
from core.api.utils.method_collections import *
from core.api.utils.checkers_collections import *
from core.api.utils.reader_collections import *
from core.api.utils.saver_collections import ResultSaver
from core.models.ecm.ErrorRunner import ErrorCorrectionModel
from core.api.utils.decorator_collections import *
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.analyzer import PerformanceAnalyzer
from core.operation.utils.utils import path_to_save_results


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

        self.YAML = YamlReader(feature_generator=self.feature_generator_dict,
                               logger=self.logger)
        self.reader = DataReader(logger=self.logger)
        self.checker = ParameterCheck(logger=self.logger)
        self.saver = ResultSaver(logger=self.logger)

    def fit(self, **kwargs):
        self._define_data()
        self.fitted_model, self.train_features = self._obtain_model(**kwargs)
        return self.fitted_model, self.train_features

    def predict(self,
                features: tuple,
                save_predictions: bool = False):
        if self.model_composer is None:
            self.logger.info(f'*------------YOU MUST FIT MODEL FIRST------------*')
        else:
            prediction = self.model_composer.predict(test_features=features)
            self.labels, self.test_features = prediction['label'], prediction['test_features']
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

        return dict(prediction=prediction_label,
                    predictions_proba=prediction_proba,
                    metrics=metrics_dict)

    def _define_data(self):
        pass

    def _obtain_model(self,
                      model_name: str,
                      task_type: str,
                      train_features: np.ndarray,
                      train_target: np.ndarray,
                      model_params: dict,
                      dataset_name: str = None,
                      ECM_mode: bool = False,
                      n_classes: int = None):
        generator_params = self.config_dict['feature_generator_params'][model_name]
        self.model_composer = self.task_pipeline_dict[task_type](generator_name=model_name,
                                                                 generator_runner=self.feature_generator_dict[
                                                                     model_name](**generator_params),
                                                                 model_hyperparams=model_params,
                                                                 ecm_model_flag=ECM_mode)
        self.model_composer.model_hyperparams['metric'] = self.checker.check_metric_type(
            n_classes=n_classes)
        self.logger.info(f'*------------{model_name} MODEL IS ON DUTY------------*')
        self.logger.info(f'*------------START FIT MODEL------------*')
        fitted_model, train_features = self.model_composer.fit(train_features=train_features,
                                                               train_target=train_target,
                                                               dataset_name=dataset_name,
                                                               baseline_type=self.config_dict['baseline'])
        return fitted_model, train_features

    def run_experiment(self, config_name):
        """Run experiment with corresponding config_name.

        Args:
            config_name: name of the config file or path to.

        """
        self.logger.info(f'*------------START EXPERIMENT------------*')
        experiment_results = {}
        self.config_dict = self.YAML.init_experiment_setup(config_name)
        for dataset_name in self.config_dict['datasets_list']:
            experiment_results[dataset_name] = {}
            experiment_dict = copy.deepcopy(self.config_dict)
            experiment_results[dataset_name]['Original'] = self._run_modelling_cycle(
                experiment_dict=experiment_dict,
                task_type=self.config_dict['task'],
                n_cycles=self.config_dict['launches'],
                dataset_name=dataset_name)

            if self.config_dict['error_correction']:
                experiment_results[dataset_name]['ECM'] = self.apply_ECM(
                    modelling_results=experiment_results[dataset_name]['Original'],
                    ECM_mode=self.config_dict['error_correction'])

            if self.config_dict['ensemble_algorithm']:
                experiment_results[dataset_name]['Ensemble'] = self.apply_ensemble(
                    modelling_results=experiment_results[dataset_name]['Original'],
                    ensemble_mode='ensemble_algorithm' in self.config_dict.keys())

            self.save_results(modelling_results=experiment_results, dataset_name=dataset_name)
        self.logger.info('*------------END OF EXPERIMENT------------*')

    def _run_modelling_cycle(self,
                             experiment_dict: dict,
                             task_type: str,
                             n_cycles: int,
                             dataset_name: str,
                             ):
        self.logger.info(f'*------------TYPE OF ML TASK - {task_type}------------*')
        modelling_results = dict()
        train_data, test_data, dataset_metainfo = self.reader.read(dataset_name=dataset_name)
        if train_data is None:
            return None

        # train_data, validation_data = self._create_validation_dataset(train_data=train_data)
        if train_data[0].shape[1] > 800:
            for exclude in ['topological', 'spectral', 'window_spectral']:
                self.logger.info(f'*------------{exclude} MODEL WAS EXCLUDE. '
                                 f'REASON: TS DATA LENGTH - {train_data[0].shape[1]}------------*')
                experiment_dict['feature_generator'].pop(exclude, None)

        for runner_name, runner in experiment_dict['feature_generator'].items():
            modelling_results[runner_name] = {}
            try:
                for launch in range(1, n_cycles + 1):
                    runner_result = {}
                    paths_to_save = os.path.join(path_to_save_results(), runner_name, dataset_name, str(launch))

                    self.logger.info(f'*------------{runner_name} MODEL IS ON DUTY------------*')
                    runner_result['test_target'] = test_data[1]
                    runner_result['path_to_save'] = paths_to_save
                    runner_result['train_target'] = train_data[1]
                    runner_result['fitted_predictor'], runner_result['train_features'] = self._obtain_model(
                        model_name=runner_name,
                        task_type=task_type,
                        train_features=train_data[0],
                        dataset_name=dataset_name,
                        train_target=train_data[1],
                        n_classes=dataset_metainfo['Number_of_classes'],
                        model_params=experiment_dict['model_params'],
                        ECM_mode=experiment_dict['error_correction'])

                    self.logger.info(f'*------------START PREDICTION AT LAUNCH-{launch}------------*')
                    label_dict = self.predict(features=test_data[0],
                                              save_predictions=True)
                    proba_dict = self.predict_proba(features=test_data[0],
                                                    save_predictions=True)
                    runner_result.update(label_dict)
                    runner_result['class_probability'] = proba_dict['class_probability']
                    runner_result['metrics'] = self.get_metrics(target=runner_result['test_target'],
                                                                prediction_proba=runner_result['class_probability'],
                                                                prediction_label=runner_result['label'])
                    self.logger.info(f'*------------METRICS AT TEST DATASET------------*')
                    metric_df = runner_result['metrics']['metrics']
                    self.logger.info(f'*-----------------------------------*')
                    self.logger.info(f'*------------{metric_df}------------*')
                    self.logger.info(f'*-----------------------------------*')
                    modelling_results[runner_name][launch] = runner_result
            except Exception as ex:
                self.logger.info(f'PROBLEM WITH {runner_name} AT LAUNCH {launch}. REASON - {ex}')
        return modelling_results

    def save_results(self, dataset_name, modelling_results: dict):
        result_at_dataset = modelling_results[dataset_name]
        for approach in result_at_dataset:
            if result_at_dataset[approach] is not None:
                for model in result_at_dataset[approach]:
                    self.saver.save_method_dict[approach](prediction=result_at_dataset[approach][model])

    def apply_ECM(self, modelling_results: Union[dict, list]):
        self.logger.info('*------------START FIT ERROR CORRECTION MODEL------------*')
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

    @Decorators.InputData
    def apply_ensemble(self,
                       modelling_results: Union[dict, list],
                       ensemble_method: str = 'AGG_voting'):
        single_mode = True
        if self.config_dict is not None:
            ensemble_method = self.config_dict['ensemble_algorithm']
            single_mode = False
        ensemble_model = self.ensemble_methods_dict[ensemble_method]()
        ensemble_results = ensemble_model.ensemble(predictions=modelling_results, single_mode=single_mode)
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
