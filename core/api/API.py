from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split
from typing import Callable
from core.api.utils.method_collections import *
from core.api.utils.checkers_collections import *
from core.api.utils.reader_collections import *
from core.api.utils.saver_collections import ResultSaver
from core.models.ecm.ErrorRunner import ErrorCorrectionModel
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import path_to_save_results


class Industrial(Fedot):
    """
    Class-support for performing experiments for tasks (read yaml configs, create data folders and log files)
    """

    def __init__(self):
        super(Fedot, self).__init__()
        self.config_dict = None
        self.logger = Logger.__call__().get_logger()
        self.model = None
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
        fitted_model, train_features = self._obtain_model(**kwargs)
        return fitted_model, train_features

    def predict(self,
                test_data: tuple,
                dataset_name: str = None):
        if self.model is None:
            self.logger.info(f'*------------YOU MUST FIT MODEL FIRST------------*')
            prediction = None
        else:
            prediction = self.model.predict(test_data, dataset_name)
        return prediction

    def plot_prediction(self, **kwargs):
        pass

    def get_metrics(self, **kwargs) -> dict:
        pass

    def _define_data(self):
        pass

    def _obtain_model(self,
                      model_name: str,
                      dataset_name: str,
                      task_type: str,
                      train_data: tuple,
                      model_params: dict,
                      ECM_mode: bool = False,
                      n_classes: int = None):

        paths_to_save = os.path.join(path_to_save_results(), model_name, dataset_name)
        self.model = self.task_pipeline_dict[task_type](generator_name=model_name,
                                                        generator_runner=self.feature_generator_dict[model_name],
                                                        model_hyperparams=model_params,
                                                        ecm_model_flag=ECM_mode)
        self.model.feature_generator_params = self.config_dict['feature_generator_params']
        self.model.model_hyperparams['metric'] = self.checker.check_metric_type(
            n_classes=n_classes)
        self.logger.info(f'*------------{model_name} MODEL IS ON DUTY------------*')
        self.logger.info(f'*------------START FIT MODEL------------*')
        fitted_model, train_features = self.model.fit(train_data, dataset_name)
        return fitted_model, train_features

    def run_experiment(self, config_name):
        """
        Run experiment with corresponding config_name
        :param config_name: configuration file name [Config_Classification.yaml]
        """
        self.logger.info(f'*------------START EXPERIMENT------------*')
        experiment_results = {}
        self.config_dict = self.YAML.init_experiment_setup(config_name)
        for dataset_name in self.config_dict['datasets_list']:
            experiment_results[dataset_name] = {}

            experiment_results[dataset_name]['Original'] = self._run_modelling_cycle(
                experiment_dict=self.config_dict,
                task_type=self.config_dict['task'],
                n_cycles=self.config_dict['launches'],
                dataset_name=dataset_name)

            experiment_results[dataset_name]['ECM'] = self._apply_ECM(
                modelling_results=experiment_results[dataset_name]['Original'],
                ECM_mode=self.config_dict['error_correction'])

            experiment_results[dataset_name]['Ensemble'] = self._apply_ensemble(
                modelling_results=experiment_results[dataset_name]['Original'],
                ensemble_mode='ensemble_algorithm' in self.config_dict.keys())

            self._save_results(dataset_name=dataset_name, modelling_results=experiment_results)
            _ = 1
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

        train_data, validation_data = self._create_validation_dataset(train_data=train_data)

        for runner_name, runner in experiment_dict['feature_generator'].items():
            modelling_results[runner_name] = {}
            try:
                for launch in range(1, n_cycles + 1):
                    runner_result = {}
                    paths_to_save = os.path.join(path_to_save_results(), runner_name, dataset_name, str(launch))

                    self.logger.info(f'*------------{runner_name} MODEL IS ON DUTY------------*')
                    runner_result['test_target'] = test_data[1]
                    runner_result['path_to_save'] = paths_to_save
                    runner_result['fitted_predictor'], runner_result['train_features'] = self._obtain_model(
                        model_name=runner_name,
                        dataset_name=dataset_name,
                        task_type=task_type,
                        train_data=train_data,
                        n_classes=dataset_metainfo['Number_of_classes'],
                        model_params=experiment_dict['model_params'],
                        ECM_mode=experiment_dict['error_correction'])

                    self.logger.info(f'*------------START PREDICTION AT LAUNCH-{launch}------------*')
                    runner_result.update(self.predict(test_data, dataset_name))
                    runner_result['predict_on_train'] = self.model.predict_on_train()
                    runner_result['predict_on_val'] = self.model.predict_on_validation(validatiom_tuple=validation_data,
                                                                                       dataset_name=dataset_name)
                    runner_result['validation_predictions'] = {f'{runner_name}_val': runner_result['predict_on_val']}
                    runner_result['test_predictions'] = {f'{runner_name}_test': runner_result['prediction']}

                    modelling_results[runner_name][launch] = runner_result
            except Exception as ex:
                self.logger.info(f'PROBLEM WITH {runner_name} AT LAUCNH {launch}. REASON - {ex}')
        return modelling_results

    def _save_results(self, dataset_name, modelling_results: dict):
        result_at_dataset = modelling_results[dataset_name]
        for approach in result_at_dataset:
            if approach is not None:
                for model in result_at_dataset[approach]:
                    self.saver.save_method_dict[approach](prediction=result_at_dataset[approach][model])

    def _apply_ECM(self, modelling_results: dict, ECM_mode: bool = False):
        if not ECM_mode:
            return None
        else:
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

    def _apply_ensemble(self, modelling_results: dict,
                        ensemble_mode: bool = False):
        if not ensemble_mode:
            return None
        else:
            ensemble_method = self.ensemble_methods_dict[self.config_dict['ensemble_algorithm']](
                train_predictions=modelling_results['train_predictions'],
                train_target=modelling_results['train_target'],
                test_target=modelling_results['test_target'])
            return ensemble_method.ensemble(predictions=modelling_results['test_predictions'])

    @staticmethod
    def _create_validation_dataset(train_data):
        X_train, X_val, y_train, y_val = train_test_split(train_data[0],
                                                          train_data[1],
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=train_data[1])
        train_data, validation_data = (X_train, y_train), (X_val, y_val)
        return train_data, validation_data
