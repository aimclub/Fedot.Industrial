from multiprocessing.dummy import Pool

import numpy as np
from fedot.api.main import Fedot
from core.spectral.SSA import Spectrum
from core.statistical.Stat_features import AggregationFeatures
from experiments.run.ExperimentRunner import ExperimentRunner
from utils.utils import *
import matplotlib.pyplot as plt
import timeit


class SSARunner(ExperimentRunner):
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None
                 ):

        super().__init__(list_of_dataset, launches, metrics_name, fedot_params)
        self.aggregator = AggregationFeatures()
        self.spectrum_extractor = Spectrum
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None

    def __vis_and_save_components(self, Components_df):
        n_rows = round(Components_df.shape[1] / 5)

        if n_rows < 4:
            plot_area = 'small'
        elif 5 < n_rows < 9:
            plot_area = 'mid'
        else:
            plot_area = 'big'

        plot_dict = {'small': (20, 10),
                     'mid': (20, 10),
                     'big': (40, 20)}

        figsize = plot_dict[plot_area]
        layout = (n_rows + 1, 5)

        Components_df.plot(subplots=True,
                           figsize=figsize,
                           legend=None,
                           layout=(n_rows + 1, 5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_save_png, f'components_for_ts_{self.count}.png'))

    def __check_Nan(self, ts):
        if any(np.isnan(ts)):
            ts = np.nan_to_num(ts, nan=0)
        return ts

    def _ts_chunk_function(self, ts):

        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')

        ts = self.__check_Nan(ts)

        spectr = self.spectrum_extractor(time_series=ts,
                                         window_length=self.window_length)
        TS_comps, X_elem, V, Components_df, _ = spectr.decompose()

        # if self.vis_flag:
        #     try:
        #         self.__vis_and_save_components(Components_df=Components_df)
        #     except Exception:
        #         self.logger.info('Vis problem')

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

    def _choose_best_window_size(self, X_train, y_train, window_length_list):
        metric_list = []
        feature_list = []

        for window_length in window_length_list:
            self.logger.info(f'Generate features for window length - {window_length}')
            self.window_length = window_length

            train_feats = self.generate_features_from_ts(X_train)

            self.logger.info(f'Validate model for window length  - {window_length}')

            metrics = self._validate_window_length(features=train_feats, target=y_train)

            self.logger.info(f'Obtained metric for window length {window_length}  - F1, ROC_AUC - {metrics}')

            metric_list.append(metrics)
            feature_list.append(train_feats)

            self.count = 0

        max_score = [sum(x) for x in metric_list]
        index_of_window = int(max(max_score))
        train_feats = feature_list[index_of_window]
        self.window_length = window_length_list[index_of_window]
        self.logger.info(f'Was choosen window length -  {self.window_length} ')
        return train_feats

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):

        self.logger.info('Generating features for fit model')
        if self.train_feats is None:
            self.train_feats = self._choose_best_window_size(X_train, y_train, window_length_list)
        self.logger.info('Start fitting FEDOT model')
        predictor = Fedot(**self.fedot_params)
        predictor.fit(features=self.train_feats, target=y_train)
        return predictor

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        self.logger.info('Generating features for prediction')

        if self.test_feats_feats is None:
            self.test_feats = self.generate_features_from_ts(ts_frame=X_test, window_length=window_length)

        start_time = timeit.default_timer()
        predictions = predictor.predict(features=self.test_feats)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=self.test_feats)
        return predictions, predictions_proba, inference
