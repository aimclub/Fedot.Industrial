from multiprocessing.dummy import Pool
from fedot.api.main import Fedot
from core.topological.TDA import Topological
from experiments.run.ExperimentRunner import ExperimentRunner
from utils.utils import *
import timeit


class TopologicalRunner(ExperimentRunner):
    def __init__(self,
                 topological_params: dict,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 ):

        super().__init__(list_of_dataset, launches, metrics_name)
        self.topological_extractor = Topological(**topological_params)

    def generate_features_from_ts(self, ts_frame, window_length=None):
        pool = Pool(8)
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]
        self.topological_extractor.time_series = ts_frame
        aggregation_df = pool.map(self.topological_extractor.time_series_rolling_betti_ripser, ts_frame.values)
        feats = pd.DataFrame(aggregation_df)
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
                          composer_params={'max_depth': 10,
                                           'max_arity': 4},
                          verbose_level=1)
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

