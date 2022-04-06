from multiprocessing.dummy import Pool

from sklearn.metrics import f1_score
from fedot.api.main import Fedot

from core.signal.wavelet import WaveletExtractor
from core.models.statistical.Stat_features import AggregationFeatures
from cases.run.ExperimentRunner import ExperimentRunner
from core.operation.utils.utils import *
import timeit


class SignalRunner(ExperimentRunner):
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None
                 ):

        super().__init__(list_of_dataset, launches, metrics_name, fedot_params)
        self.aggregator = AggregationFeatures()
        self.wavelet_extractor = WaveletExtractor
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None

    def __check_Nan(self, ts):
        if any(np.isnan(ts)):
            ts = np.nan_to_num(ts, nan=0)
        return ts

    def _ts_chunk_function(self, ts):

        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')

        ts = self.__check_Nan(ts)
        features = []
        spectr = self.wavelet_extractor(time_series=ts, wavelet_name=self.wavelet)
        high_freq, low_freq = spectr.decompose_signal()

        for mph in range(3):
            peaks_high_freq = spectr.detect_peaks(high_freq, mph=mph + 1)
            features.append(len(peaks_high_freq))
            peaks_high_freq = spectr.detect_peaks(high_freq, threshold=mph + 1, valley=True)
            features.append(len(peaks_high_freq))
            low_freq_freq = spectr.detect_peaks(low_freq, mph=mph + 1)
            features.append(len(low_freq_freq))
            low_freq_freq = spectr.detect_peaks(low_freq, threshold=mph + 1, valley=True)
            features.append(len(low_freq_freq))

        for mpd in [1, 3, 5, 7, 9]:
            peaks_high_freq = spectr.detect_peaks(high_freq, mpd=mpd)
            low_freq_freq = spectr.detect_peaks(low_freq, mpd=mpd)
            features.append(len(peaks_high_freq))
            features.append(len(low_freq_freq))

        self.count += 1
        return features

    def generate_vector_from_ts(self, ts_frame):
        pool = Pool(8)
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]
        components_and_vectors = pool.map(self._ts_chunk_function, ts_frame.values)
        pool.close()
        pool.join()
        self.logger.info(f'Time spent on wavelet transformation - {timeit.default_timer() - start}')
        return components_and_vectors

    def generate_features_from_ts(self, eigenvectors_list):
        pool = Pool(8)
        start = timeit.default_timer()
        aggregation_df = pool.map(self.aggregator.create_features,
                                  eigenvectors_list)
        pool.close()
        pool.join()
        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return aggregation_df

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

    def _choose_best_wavelet(self, X_train, y_train, wavelet_list):

        metric_list = []
        feature_list = []

        for wavelet in wavelet_list:
            self.logger.info(f'Generate features for window length - {wavelet}')
            self.wavelet = wavelet

            features = self.generate_vector_from_ts(X_train)
            train_feats = pd.DataFrame(features)

            self.logger.info(f'Validate model for wavelet  - {wavelet}')

            metrics = self._validate_window_length(features=train_feats, target=y_train)

            self.logger.info(f'Obtained metric for wavelet {wavelet}  - F1, ROC_AUC - {metrics}')

            metric_list.append(metrics)
            feature_list.append(train_feats)
            self.count = 0

        max_score = [sum(x) for x in metric_list]
        index_of_window = int(max_score.index(max(max_score)))
        train_feats = feature_list[index_of_window]

        self.wavelet = wavelet_list[index_of_window]
        self.logger.info(f'Was choosen wavelet -  {self.wavelet} ')

        return train_feats

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):

        self.logger.info('Generating features for fit model')
        if self.train_feats is None:
            self.train_feats = self._choose_best_wavelet(X_train, y_train, window_length_list)
        self.logger.info('Start fitting FEDOT model')
        predictor = Fedot(**self.fedot_params)

        if self.fedot_params['composer_params']['metric'] == 'f1':
            predictor.params.api_params['tuner_metric'] = f1_score

        predictor.fit(features=self.train_feats, target=y_train)
        return predictor

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        self.logger.info('Generating features for prediction')

        if self.test_feats is None:
            features = self.generate_vector_from_ts(X_test)
            self.test_feats = pd.DataFrame(features)

        start_time = timeit.default_timer()
        predictions = predictor.predict(features=self.test_feats)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=self.test_feats)
        return predictions, predictions_proba, inference
