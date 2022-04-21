import json
import os.path
# import pandas as pd
from core.metrics.metrics_implementation import *
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from core.operation.utils.booster import Booster
from core.operation.utils.static_booster import StaticBooster
from sklearn.model_selection import train_test_split
from cases.analyzer import PerfomanceAnalyzer
from core.operation.utils.utils import *
from cases.run.utils import *

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
                 boost_mode: bool = True):
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name
        self.count = 0
        self.window_length = None
        self.logger = get_logger()
        self.fedot_params = fedot_params
        self.boost_mode = boost_mode
        self.y_test = None

    def generate_features_from_ts(self, ts_frame, window_length=None):
        """  Method responsible for  experiment pipeline """
        return

    def _generate_fit_time(self, predictor):
        """  Method responsible for  experiment pipeline """
        return

    def _create_path_to_save(self, dataset, launch):
        """  Method responsible for  experiment pipeline """
        return

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
        """  Method responsible for  experiment pipeline """
        return

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
        metric_roc = ROCAUC()

        score_f1 = metric_f1.metric(target=prediction.target, prediction=prediction.predict)

        try:
            try:
                score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
            except Exception:
                prediction = pipeline.predict(input_data=test_data, output_mode='probs')
                score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
        except Exception:
            score_roc_auc = 0.5

        return score_f1, score_roc_auc

    def run_experiment(self,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict):

        for dataset in self.list_of_dataset:

            self.train_feats = None
            self.test_feats = None

            for launch in range(self.launches):
                try:
                    self.path_to_save = self._create_path_to_save(dataset, launch)
                    self.path_to_save_png = os.path.join(self.path_to_save, 'pictures')

                    if not os.path.exists(self.path_to_save_png):
                        os.makedirs(self.path_to_save_png)

                    X, y = dict_of_dataset[dataset]
                    if type(X) is tuple:
                        X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))

                    n_classes = np.unique(y_train)

                    if n_classes.shape[0] > 2:
                        self.fedot_params['composer_params']['metric'] = 'f1'
                    else:
                        self.fedot_params['composer_params']['metric'] = 'roc_auc'

                    # self.train_feats

                    predictor = self.fit(X_train=X_train,
                                         y_train=y_train,
                                         window_length_list=dict_of_win_list[dataset])

                    self.count = 0
                    self.test_feats = self.train_feats

                    # Predict on whole TRAIN
                    predictions_train, predictions_proba_train, _ = self.predict(predictor=predictor,
                                                                                 X_test=X_train,
                                                                                 window_length=self.window_length,
                                                                                 y_test=y_train)

                    # GEt metrics on whole TRAIN
                    try:
                        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                                  target=y_test,
                                                                  predicted_labels=predictions_train,
                                                                  predicted_probs=predictions_proba_train
                                                                  )
                    except Exception as ex:
                        metrics = 'empty'

                    self.logger.info(f'Without Boosting metrics are: {metrics}')

                    if n_classes.shape[0] > 2:
                        base_predict = predictions_train.reshape(-1)
                    else:
                        base_predict = predictions_proba_train.reshape(-1)

                    booster = Booster(X_train=self.train_feats,
                                      y_train=y_train,
                                      base_predict=base_predict,
                                      # timeout=round(self.fedot_params['timeout']/2),
                                      timeout=4)

                    predictions_boosting_train, model_list, ensemble_model = booster.run_boosting()
                    # Predict on whole TEST and generate self.test_features

                    self.test_feats = None

                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test,
                                                                             window_length=self.window_length,
                                                                             y_test=y_test)

                    boosting_test = []

                    for model in model_list:
                        boosting_test.append(model.predict(self.test_feats))
                    boosting_test = [x.reshape(-1) for x in boosting_test]
                    error_correction = ensemble_model.predict(pd.DataFrame(data=boosting_test).T)
                    corrected_probs = error_correction.reshape(-1) + predictions_proba.reshape(-1)
                    corrected_probs = corrected_probs.reshape(-1)
                    corrected_labels = abs(np.round(corrected_probs))

                    solution_table = pd.DataFrame({'target': y_test,
                                                   'base_probs': predictions_proba.reshape(-1),
                                                   'ensemble': error_correction.reshape(-1),
                                                   'corrected_probs': corrected_probs,
                                                   'corrected_labels': corrected_labels})
                    for index, data in enumerate(boosting_test):
                        solution_table[f'boost_{index + 1}'] = data.reshape(-1)

                    # GEt metrics on whole TEST

                    try:
                        metrics_boosting = self.analyzer.calculate_metrics(self.metrics_name,
                                                                           target=y_test,
                                                                           predicted_labels=corrected_labels,
                                                                           predicted_probs=corrected_probs
                                                                           )
                        metrics_without_boosting = self.analyzer.calculate_metrics(self.metrics_name,
                                                                                   target=y_test,
                                                                                   predicted_labels=predictions,
                                                                                   predicted_probs=predictions_proba
                                                                                   )

                        no_boost = pd.Series(metrics_without_boosting)
                        boost = pd.Series(metrics_boosting)
                        metrics_table = pd.concat([no_boost, boost], axis=1).reset_index()
                        metrics_table.columns = ['metric', 'before_boosting', 'after_boosting']

                    except Exception as ex:
                        metrics_boosting = 'empty'

                    self.save_boosting_results(dataset=dataset,
                                               solution_table=solution_table,
                                               metrics_table=metrics_table,
                                               model_list=model_list,
                                               ensemble_model=ensemble_model)

                    self.logger.info('Saving model')
                    predictor.current_pipeline.save(path=self.path_to_save)
                    best_pipeline, fitted_operation = predictor.current_pipeline.save()
                    try:
                        opt_history = predictor.history.save()
                        with open(os.path.join(self.path_to_save, 'history', 'opt_history.json'), 'w') as f:
                            json.dump(json.loads(opt_history), f)
                    except Exception as ex:
                        ex = 1

                    self.logger.info('Saving results')

                    save_results(predictions=predictions,
                                 prediction_proba=predictions_proba,
                                 target=y_test,
                                 metrics=metrics_boosting,
                                 inference=inference,
                                 fit_time=np.mean(self._generate_fit_time(predictor)),
                                 path_to_save=self.path_to_save,
                                 window=self.window_length)
                    self.count = 0

                except Exception as ex:
                    self.count = 0
                    print(ex)
                    print(str(dataset))

    @staticmethod
    def save_boosting_results(dataset, solution_table, metrics_table, model_list, ensemble_model):
        location = os.path.join(project_path(), 'results_of_experiments', dataset, 'boosting')
        if not os.path.exists(location):
            os.makedirs(location)
        solution_table.to_csv(os.path.join(location, 'solution_table.csv'))
        metrics_table.to_csv(os.path.join(location, 'metrics_table.csv'))

        models_path = os.path.join(location, 'boosting_pipelines')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for index, model in enumerate(model_list):
            model.current_pipeline.save(path=os.path.join(models_path, f'boost_{index}'),
                                        datetime_in_path=False)
        ensemble_model.current_pipeline.save(path=os.path.join(models_path, 'boost_ensemble'),
                                             datetime_in_path=False)
