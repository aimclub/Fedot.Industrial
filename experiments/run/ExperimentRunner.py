import json
import os

import numpy as np

from metrics.metrics_implementation import *
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
from experiments.analyzer import PerfomanceAnalyzer
from core.spectral.SSA import Spectrum
from core.statistical.Stat_features import AggregationFeatures
from utils.utils import *
from experiments.run.utils import *

dict_of_dataset = dict
dict_of_win_list = dict


class ExperimentRunner:
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = {'problem': 'classification',
                                       'seed': 42,
                                       'timeout': 10,
                                       'composer_params': {'max_depth': 10,
                                                           'max_arity': 4},
                                       'verbose_level': 1}):
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name
        self.count = 0
        self.window_length = None
        self.logger = get_logger()
        self.fedot_params = fedot_params

    def generate_features_from_ts(self, ts_frame, window_length=None):
        """  Method responsible for  experiment pipeline """
        return

    def _generate_fit_time(self, predictor):
        """  Method responsible for  experiment pipeline """
        return

    def _create_path_to_save(self, dataset, launch):
        """  Method responsible for  experiment pipeline """
        return

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):
        """  Method responsible for  experiment pipeline """
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
            score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
        except Exception:
            prediction = pipeline.predict(input_data=test_data, output_mode='probs')
            score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)

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

                    predictor = self.fit(X_train=X_train,
                                         y_train=y_train,
                                         window_length_list=dict_of_win_list[dataset])

                    self.count = 0

                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test,
                                                                             window_length=self.window_length,
                                                                             y_test=None)
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
                    try:
                        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                                  target=y_test,
                                                                  predicted_labels=predictions,
                                                                  predicted_probs=predictions_proba)
                    except Exception as ex:
                        metrics = 'empty'

                    save_results(predictions=predictions,
                                 prediction_proba=predictions_proba,
                                 target=y_test,
                                 metrics=metrics,
                                 inference=inference,
                                 fit_time=np.mean(self._generate_fit_time(predictor)),
                                 path_to_save=self.path_to_save)
                    self.count = 0

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
