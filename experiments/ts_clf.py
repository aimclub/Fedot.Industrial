import json
import logging
import multiprocessing as mp  # Организует параллельность
from multiprocessing.dummy import Pool

from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split
from sktime.datasets import *
from experiments.analyzer import PerfomanceAnalyzer
from core.spectral.SSA import Spectrum
from core.statistical.Stat_features import AggregationFeatures
from utils.utils import *
import timeit
from tqdm import tqdm
import logging


def read_tsv(file_name: str):
    df_train = pd.read_csv(
        'D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\IndustrialTS\data\{}\{}_TRAIN.tsv'.format(file_name, file_name),
        sep='\t',
        header=None)
    X_train = df_train.iloc[:, 1:]
    y_train = df_train[0].values
    df_test = pd.read_csv(
        'D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\IndustrialTS\data\{}\{}_TEST.tsv'.format(file_name, file_name),
        sep='\t',
        header=None)
    X_test = df_test.iloc[:, 1:]
    y_test = df_test[0].values

    return (X_train, X_test), (y_train, y_test)


dict_of_dataset = {'gunpoint': load_gunpoint(return_X_y=True),
                   'basic_motions': load_basic_motions(return_X_y=True),
                   'arrow_head': load_arrow_head(return_X_y=True),
                   'osuleaf': load_osuleaf(return_X_y=True),
                   'italy_power': load_italy_power_demand(return_X_y=True),
                   'unit_test': load_unit_test(return_X_y=True),
                   'Herring': read_tsv('Herring'),
                   'Haptics': read_tsv('Haptics'),
                   'DodgerLoopDay': read_tsv('DodgerLoopDay'),
                   'Earthquakes': read_tsv('Earthquakes'),
                   'FordA': read_tsv('FordA'),
                   'FordB': read_tsv('FordB'),
                   'Plane': read_tsv('Plane'),
                   'Trace': read_tsv('Trace')
                   }

dict_of_win_list = {'gunpoint': 30,
                    'basic_motions': 10,
                    'arrow_head': 50,
                    'osuleaf': 90,
                    'italy_power': 3,
                    'unit_test': 3,
                    'Herring': 170,
                    'Haptics': 300,
                    'DodgerLoopDay': 80,
                    'Earthquakes': 128,
                    'FordA': 125,
                    'FordB': 125,
                    'Plane': 48,
                    'Trace': 90
                    }


class ExperimentRunner:
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 1,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']):
        self.aggregator = AggregationFeatures()
        self.spectrum_extractor = Spectrum
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name
        self.count = 0
        logger = logging.getLogger('Experiment logger')
        logger.setLevel(logging.INFO)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        self.logger = logger

    def _ts_chunk_function(self, ts):

        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')
        spectr = self.spectrum_extractor(time_series=ts,
                                         window_length=self.window_length)
        TS_comps, X_elem, V, Components_df, _ = spectr.decompose()
        aggregation_df = self.aggregator.create_features(feature_to_aggregation=Components_df.iloc[:, :10])
        self.count += 1
        return aggregation_df

    def generate_features_from_ts(self, ts_frame, window_length=None):
        pool = Pool(8)
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]
        aggregation_df = pool.map(self._ts_chunk_function, ts_frame.values)
        feats = pd.concat(aggregation_df)
        pool.close()
        pool.join()
        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return feats

    def _generate_fit_time(self, predictor):
        fit_time = []
        if predictor.best_models is None:
            fit_time.append(predictor.current_pipeline.computation_time)
        else:
            for model in predictor.best_models:
                current_computation = model.computation_time
                fit_time.append(current_computation)
        return fit_time

    def _create_path_to_save(self, dataset, launch):
        save_path = os.path.join(path_to_save_results(), dataset, str(launch))
        return save_path

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length: int = None):

        self.logger.info('Generating features for fit model')
        self.window_length = window_length
        train_feats = self.generate_features_from_ts(X_train)
        self.logger.info('Start fitting FEDOT model')
        predictor = Fedot(problem='classification',
                          seed=42,
                          timeout=10,
                          composer_params={'max_depth':10,
                                           'max_arity':4},
                          verbose_level=4)
        predictor.fit(features=train_feats, target=y_train)
        return predictor

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None):
        self.logger.info('Generating features for prediction')
        test_feats = self.generate_features_from_ts(ts_frame=X_test, window_length=window_length)
        start_time = timeit.default_timer()
        predictions = predictor.predict(features=test_feats)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=test_feats)
        return predictions, predictions_proba, inference

    def run_experiment(self):
        prediction_list = []
        target = []
        models = []
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
                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test,
                                                                             window_length=dict_of_win_list[dataset])
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
                    models.append(predictor)

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
        return models


if __name__ == '__main__':
    list_of_dataset = [ #'FordA',
        #'FordB',
        #'DodgerLoopDay',
        'Earthquakes',
        #'Plane',
        #'Trace'
    ]
    runner = ExperimentRunner(list_of_dataset)
    models = runner.run_experiment()
    _ = 1
