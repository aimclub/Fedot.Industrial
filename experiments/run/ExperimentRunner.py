import json
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

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length: int = None):
        """  Method responsible for  experiment pipeline """
        return

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        """  Method responsible for  experiment pipeline """
        return

    def run_experiment(self,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict):
        for dataset in self.list_of_dataset:
            for launch in range(self.launches):
                try:
                    path_to_save = self._create_path_to_save(dataset, launch)
                    X, y = dict_of_dataset[dataset]
                    if type(X) is tuple:
                        X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))

                    predictor = self.fit(X_train=X_train,
                                         y_train=y_train,
                                         window_length=dict_of_win_list[dataset])

                    self.count = 0

                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test,
                                                                             window_length=dict_of_win_list[dataset],
                                                                             y_test=None)
                    self.logger.info('Saving model')
                    predictor.current_pipeline.save(path=path_to_save)
                    best_pipeline, fitted_operation = predictor.current_pipeline.save()
                    try:
                        opt_history = predictor.history.save()
                        with open(os.path.join(path_to_save, 'history', 'opt_history.json'), 'w') as f:
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
                                 path_to_save=path_to_save)
                    self.count = 0

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
