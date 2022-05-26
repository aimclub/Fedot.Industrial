import json
import os.path

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split

from cases.analyzer import PerfomanceAnalyzer
from cases.run.utils import *
from core.metrics.metrics_implementation import *
from core.operation.utils.Decorators import exception_decorator
from core.operation.utils.booster import Booster
from core.operation.utils.static_booster import StaticBooster
from core.operation.utils.utils import *

dict_of_dataset = dict
dict_of_win_list = dict


class ExperimentRunner:
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = {'problem': 'classification',
                                       'seed': 42,
                                       'timeout': 1,
                                       'composer_params': {'max_depth': 10,
                                                           'max_arity': 4},
                                       'verbose_level': 1},
                 boost_mode: bool = True,
                 static_booster: bool = False):
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name
        self.count = 0
        self.window_length = None
        self.logger = get_logger()
        self.fedot_params = fedot_params
        self.boost_mode = boost_mode

        self.static_booster = static_booster
        self.y_test = None

    def check_Nan(self, ts):
        if any(np.isnan(ts)):
            ts = np.nan_to_num(ts, nan=0)
        return ts

    def generate_features_from_ts(self, ts_frame, window_length=None):
        """  Method responsible for  experiment pipeline """
        return

    def generate_fit_time(self, predictor):
        fit_time = []
        if predictor.best_models is None:
            fit_time.append(predictor.current_pipeline.computation_time)
        else:
            for model in predictor.best_models:
                current_computation = model.computation_time
                fit_time.append(current_computation)
        return fit_time

    def _create_path_to_save(self, method, dataset, launch):
        save_path = os.path.join(path_to_save_results(), method, dataset, str(launch))
        return save_path

    def _load_data(self, dataset, dict_of_dataset):
        X, y = dict_of_dataset[dataset]
        if type(X) is tuple:
            X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):
        """  Method responsible for  experiment pipeline """
        return

    def extract_features(self,
                         dataset_name: str,
                         dict_of_dataset: dict = None,
                         dict_of_extra_params: dict = None):
        return

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        """  Method responsible for experiment pipeline """
        return

    def _get_clf_params(self):
        self.n_classes = np.unique(self.y_train).shape[0]

        if self.n_classes > 2:
            self.fedot_params['composer_params']['metric'] = 'f1'
        else:
            self.fedot_params['composer_params']['metric'] = 'roc_auc'

    def _get_dimension_params(self, predictions_proba_train):
        if self.n_classes > 2:
            self.base_predict = predictions_proba_train
            y_train_multi = pd.get_dummies(self.y_train, sparse=True)
            self.y_train = y_train_multi.values
            self.reshape_flag = False
        else:
            self.base_predict = predictions_proba_train.reshape(-1)
            self.base_predict = round(np.max(self.y_test) - np.max(predictions_proba_train)) + self.base_predict
            self.reshape_flag = True

    def _convert_boosting_prediction(self, boosting_stages_predict, ensemble_model, predictions_proba):
        self.logger.info('Calculation of error correction by boosting')
        if self.static_booster:
            task = Task(TaskTypesEnum.regression)
            input_data = InputData(idx=np.arange(0, len(boosting_stages_predict)),
                                   features=boosting_stages_predict,
                                   target=self.y_test,
                                   task=task,
                                   data_type=DataTypesEnum.table)
        else:
            input_data = boosting_stages_predict

        if self.reshape_flag:
            # boosting_stages_predict = [x.reshape(-1) for x in boosting_stages_predict]
            if self.static_booster:
                error_correction = ensemble_model.predict(input_data).predict.reshape(-1)
            else:
                error_correction = ensemble_model.predict(input_data).reshape(-1)

            corrected_probs = error_correction.reshape(-1) + predictions_proba.reshape(-1)
            if self.reshape_flag and np.min(self.y_test) == 1:
                corrected_probs = (error_correction.reshape(-1) + predictions_proba.reshape(-1)) / 2
            corrected_probs = corrected_probs.reshape(-1)
            corrected_labels = abs(np.round(corrected_probs))

        else:
            # error_correction = ensemble_model.predict(np.hstack(boosting_test))
            dictlist = []
            for value in boosting_stages_predict:
                dictlist.append(value)
            error_correction = sum(dictlist)
            corrected_probs = error_correction + predictions_proba
            corrected_labels = np.array([x.argmax() + min(self.y_test) for x in corrected_probs])
            corrected_labels = corrected_labels.reshape(len(corrected_labels), 1)

        return dict(corrected_labels=corrected_labels,
                    corrected_probs=corrected_probs,
                    error_correction=error_correction)

    def _validate_window_length(self, features, target):

        node = PrimaryNode('rf')
        pipeline = Pipeline(node)
        n_samples = round(features.shape[0] * 0.7)

        train_data = InputData(features=features.values[:n_samples, :], target=target[:n_samples],
                               idx=np.arange(0, len(target[:n_samples])),
                               task=Task(TaskTypesEnum('classification')), data_type=DataTypesEnum.table)
        test_data = InputData(features=features.values[n_samples:, :], target=target[n_samples:],
                              idx=np.arange(0, len(target[n_samples:])),
                              task=Task(TaskTypesEnum('classification')), data_type=DataTypesEnum.table)

        fitted = pipeline.fit(input_data=train_data)
        prediction = pipeline.predict(input_data=test_data, output_mode='labels')
        metric_f1 = F1()
        score_f1 = metric_f1.metric(target=prediction.target, prediction=prediction.predict)

        score_roc_auc = self.get_roc_auc_score(pipeline, prediction, test_data)
        if score_roc_auc is None:
            score_roc_auc = 0.5

        return score_f1, score_roc_auc

    @exception_decorator(exception_return=0.5)
    def get_roc_auc_score(self, pipeline, prediction, test_data):
        metric_roc = ROCAUC()
        try:
            score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
        except Exception as error:
            prediction = pipeline.predict(input_data=test_data, output_mode='probs')
            score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
        return score_roc_auc

    def _predict_on_train(self, predictor):

        self.count = 0
        self.test_feats = self.train_feats

        # Predict on whole TRAIN
        predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                 X_test=self.X_train,
                                                                 window_length=self.window_length,
                                                                 y_test=self.y_train)
        # if self.n_classes == 2:
        #     predictions = self.proba_to_vector(predictions)
        #     predictions_proba = self.proba_to_vector(predictions_proba)

        # GEt metrics on TRAIN
        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                  target=self.y_train,
                                                  predicted_labels=predictions,
                                                  predicted_probs=predictions_proba
                                                  )
        return dict(predictions=predictions,
                    predictions_proba=predictions_proba,
                    inference=inference,
                    metrics=metrics)

    def _predict_on_test(self, predictor):

        self.test_feats = None
        predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                 X_test=self.X_test,
                                                                 window_length=self.window_length,
                                                                 y_test=self.y_test)
        #  make predictions/predictions_proba conversion to vector IF binary classification
        # if self.n_classes == 2:
        #     predictions = self.proba_to_vector(predictions)
        #     predictions_proba = self.proba_to_vector(predictions_proba)

        # GEt metrics on whole TEST

        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                  target=self.y_test,
                                                  predicted_labels=predictions,
                                                  predicted_probs=predictions_proba
                                                  )
        self.logger.info(f'Without Boosting metrics are: {metrics}')
        result_on_test = dict(predictions=predictions,
                              predictions_proba=predictions_proba,
                              inference=inference,
                              metrics=metrics)
        if self.window_length is None:
            self.window_length = 'Empty'
        result_on_test['window'] = self.window_length
        result_on_test['target'] = self.y_test
        result_on_test['fit_time'] = self._generate_fit_time(predictor=predictor)[0]
        result_on_test['path_to_save'] = self.path_to_save
        if self.reshape_flag and np.min(self.y_test) == 1:
            result_on_test['predictions_proba'] = round(
                np.max(self.y_test) - np.max(result_on_test['predictions_proba'])) + \
                                                  result_on_test['predictions_proba']

        return result_on_test

    def proba_to_vector(self, matrix, dummy_flag=True):

        if not dummy_flag:
            dummy_val = -1
        else:
            dummy_val = 0
        if len(matrix.shape) > 1:
            vector = np.array([x.argmax() + x[x.argmax()] + dummy_val for x in matrix])
            return vector
        return matrix

    def static_boosting_pipeline(self, predictions_proba, model_list, ensemble_model):
        boosting_stages_predict = []
        task = Task(TaskTypesEnum.regression)
        input_data_test = InputData(idx=np.arange(0, len(self.test_feats)),
                                    features=self.test_feats,
                                    target=self.y_test,
                                    task=task,
                                    data_type=DataTypesEnum.table)

        for model in model_list:
            boost_predict = model.predict(input_data=input_data_test).predict.reshape(-1)
            boosting_stages_predict.append(boost_predict)

        boosting_stages_predict = pd.DataFrame([x.reshape(-1) for x in boosting_stages_predict]).T

        input_data = InputData(idx=np.arange(0, len(boosting_stages_predict)),
                               features=boosting_stages_predict,
                               target=self.y_test,
                               task=task,
                               data_type=DataTypesEnum.table)

        error_correction = ensemble_model.predict(input_data=input_data).predict.reshape(-1)
        boosting_result = self._convert_boosting_prediction(boosting_stages_predict=boosting_stages_predict,
                                                            ensemble_model=ensemble_model,
                                                            predictions_proba=predictions_proba)
        return boosting_result

    def genetic_boosting_pipeline(self, predictions_proba, model_list, ensemble_model) -> dict:
        self.logger.info('Predict on booster models')
        boosting_stages_predict = []
        input_data_test = self.test_feats

        n = 1
        for model in model_list:
            self.logger.info(f'Cycle {n} of boosting has started')
            boost_predict = model.predict(input_data_test)
            boosting_stages_predict.append(boost_predict)
            n += 1

        self.logger.info('Ensebling booster predictions')
        if ensemble_model:
            self.logger.info('Ensembling using FEDOT has been chosen')
            boosting_stages_predict = pd.DataFrame(i.reshape(-1) for i in boosting_stages_predict).T
        else:
            boosting_stages_predict = [np.array(_) for _ in boosting_stages_predict]
            self.logger.info('Ensembling using SUM method has been chosen')

        boosting_result = self._convert_boosting_prediction(boosting_stages_predict=boosting_stages_predict,
                                                            ensemble_model=ensemble_model,
                                                            predictions_proba=predictions_proba)
        return boosting_result

    def _predict_with_boosting(self,
                               predictions,
                               predictions_proba,
                               metrics_without_boosting):

        if self.static_booster:
            booster = StaticBooster(X_train=self.train_feats,
                                    y_train=self.y_train,
                                    base_predict=self.base_predict,
                                    timeout=round(self.fedot_params['timeout'] / 2),
                                    )
            boosting_pipeline = self.static_boosting_pipeline
        else:
            booster = Booster(X_train=self.train_feats,
                              y_train=self.y_train,
                              base_predict=self.base_predict,
                              timeout=round(self.fedot_params['timeout'] / 2),
                              reshape_flag=self.reshape_flag
                              )
            boosting_pipeline = self.genetic_boosting_pipeline

        predictions_boosting_train, model_list, ensemble_model = booster.run_boosting()
        results_on_test = boosting_pipeline(predictions_proba,
                                            model_list,
                                            ensemble_model)

        results_on_test['base_probs_on_test'] = predictions_proba
        # results_on_test['true_labels_on_test'] = predictions

        solution_table = pd.DataFrame({'target': self.y_test,
                                       'base_probs_on_test': self.proba_to_vector(
                                           results_on_test['base_probs_on_test']),
                                       'corrected_probs': self.proba_to_vector(results_on_test['corrected_probs']),
                                       'corrected_labels': self.proba_to_vector(results_on_test['corrected_labels'])})

        metrics_boosting = self.analyzer.calculate_metrics(self.metrics_name,
                                                           target=self.y_test,
                                                           predicted_labels=results_on_test['corrected_labels'],
                                                           predicted_probs=results_on_test['corrected_probs']
                                                           )

        no_boost = pd.Series(metrics_without_boosting)
        boost = pd.Series(metrics_boosting)
        metrics_table = pd.concat([no_boost, boost], axis=1).reset_index()
        metrics_table.columns = ['metric', 'before_boosting', 'after_boosting']

        return dict(solution_table=solution_table,
                    metrics_table=metrics_table,
                    model_list=model_list,
                    ensemble_model=ensemble_model)

    @exception_decorator(exception_return=1)
    def _save_all_results(self, predictor, boosting_results, normal_results, save_boosting=False):

        self.logger.info('Saving results')
        if save_boosting:
            self.save_boosting_results(**boosting_results)
        else:
            predictor.current_pipeline.save(path=self.path_to_save)
            best_pipeline, fitted_operation = predictor.current_pipeline.save()
            opt_history = predictor.history.save()

            # history_path = os.path.join(self.path_to_save, 'history')
            # if not os.path.exists(history_path):
            #     os.makedirs(history_path)
            with open(os.path.join(self.path_to_save, 'opt_history.json'), 'w') as f:
                json.dump(json.loads(opt_history), f)

            save_results(**normal_results)

    def run_experiment(self,
                       method: str,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict,
                       save_features=False,
                       single_window_mode=False):

        for dataset in self.list_of_dataset:
            self.train_feats = None
            self.test_feats = None

            self.launches_run(method=method,
                              dataset=dataset,
                              dict_of_dataset=dict_of_dataset,
                              dict_of_win_list=dict_of_win_list,
                              save_features=save_features,
                              single_window_mode=single_window_mode)

    @exception_decorator(exception_return='Problem')
    def get_ECM_results(self, result_on_test, dataset, predictor):
        boosting_results = self._predict_with_boosting(predictions=result_on_test['predictions'],
                                                       predictions_proba=result_on_test['predictions_proba'],
                                                       metrics_without_boosting=result_on_test['metrics'])
        boosting_results['dataset'] = dataset

        self._save_all_results(predictor=predictor,
                               boosting_results=boosting_results,
                               normal_results=result_on_test,
                               save_boosting=True)

    def launches_run(self, method, dataset, dict_of_dataset, dict_of_win_list, save_features=False,
                     single_window_mode=False):
        trajectory_windows_list = dict_of_win_list[dataset]
        for launch in range(self.launches):
            self.path_to_save = self._create_path_to_save(method, dataset, launch)
            self.path_to_save_png = os.path.join(self.path_to_save, 'pictures')

            if not os.path.exists(self.path_to_save_png):
                os.makedirs(self.path_to_save_png)

            X, y = dict_of_dataset[dataset]
            self.X_train, self.X_test, self.y_train, self.y_test = X[0], X[1], y[0], y[1]

            self._get_clf_params()

            if single_window_mode:
                self.window_length = trajectory_windows_list[launch]
                window_length_list = trajectory_windows_list[launch]
                self.logger.info(
                    'Generate pipeline for trajectory matrix with window length - {}'.format(self.window_length))
                self.test_feats = None
                self.train_feats = None
            else:
                window_length_list = trajectory_windows_list

            predictor = self.fit(X_train=self.X_train,
                                 y_train=self.y_train,
                                 window_length_list=window_length_list)

            result_on_train = self._predict_on_train(predictor=predictor)

            self._get_dimension_params(predictions_proba_train=result_on_train['predictions_proba'])

            result_on_test = self._predict_on_test(predictor=predictor)

            if save_features:
                pd.DataFrame(self.train_feats).to_csv(os.path.join(self.path_to_save, 'train_features.csv'))
                pd.DataFrame(self.y_train).to_csv(os.path.join(self.path_to_save, 'train_target.csv'))
                pd.DataFrame(self.test_feats).to_csv(os.path.join(self.path_to_save, 'test_features.csv'))
                pd.DataFrame(self.y_test).to_csv(os.path.join(self.path_to_save, 'test_target.csv'))

            self._save_all_results(predictor=predictor,
                                   boosting_results=None,
                                   normal_results=result_on_test)

            self.get_ECM_results(result_on_test, dataset, predictor)

    def save_boosting_results(self, dataset, solution_table, metrics_table, model_list, ensemble_model):
        location = os.path.join(self.path_to_save, 'boosting')
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
            logger = get_logger()
            logger.info('Ensemble model cannot be saved due to applied SUM method ')
