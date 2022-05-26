import pandas as pd
from sklearn.metrics import f1_score
from fedot.api.main import Fedot
from core.models.statistical.Stat_features import AggregationFeatures
from cases.run.ExperimentRunner import ExperimentRunner
from core.operation.utils.utils import *
import timeit


class StatsRunner(ExperimentRunner):
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None,
                 static_booster: bool = False,
                 window_mode: bool = False
                 ):

        super().__init__(list_of_dataset, launches, metrics_name, fedot_params, static_booster=static_booster)
        self.aggregator = AggregationFeatures()
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def generate_features_from_ts(self, ts):
        self.ts_samples_count = ts.shape[0]
        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')
        start = timeit.default_timer()
        ts = self.check_Nan(ts)
        ts = pd.DataFrame(ts, dtype=float)

        if self.window_mode:
            list_with_stat_features_on_interval = apply_window_for_statistical_feature(ts_data=ts,
                                                                                       feature_generator=self.aggregator.create_baseline_features)
            aggregation_df = pd.concat(list_with_stat_features_on_interval, axis=1)
        else:
            aggregation_df = self.aggregator.create_baseline_features(ts)
        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return aggregation_df

    def extract_features(self,
                         dataset,
                         dict_of_dataset,
                         dict_of_extra_params=None):
        X_train, X_test, y_train, y_test = self._load_data(dataset=dataset, dict_of_dataset=dict_of_dataset)
        return self.generate_features_from_ts(X_train), pd.DataFrame(self.generate_features_from_ts(X_test))

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):

        self.logger.info('Generating features for fit model')
        if self.train_feats is None:
            self.train_feats = self.generate_features_from_ts(X_train)
            self.train_feats = delete_col_by_var(self.train_feats)
        self.logger.info('Start fitting FEDOT model')
        self.fedot_params['safe_mode'] = False
        predictor = Fedot(**self.fedot_params)

        if self.fedot_params['composer_params']['metric'] == 'f1':
            predictor.params.api_params['tuner_metric'] = f1_score

        predictor.fit(features=self.train_feats, target=y_train)
        return predictor

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        self.logger.info('Generating features for prediction')

        if self.test_feats is None:
            features = self.generate_features_from_ts(X_test)
            self.test_feats = pd.DataFrame(features[self.train_feats.columns])

        start_time = timeit.default_timer()
        predictions = predictor.predict(features=self.test_feats)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=self.test_feats)
        return predictions, predictions_proba, inference
