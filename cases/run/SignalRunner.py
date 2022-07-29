import timeit

from tqdm import tqdm

from cases.run.ExperimentRunner import ExperimentRunner
from cases.run.utils import read_tsv
from core.models.signal.wavelet import WaveletExtractor
from core.models.statistical.Stat_features import AggregationFeatures
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import *


class SignalRunner(ExperimentRunner):
    def __init__(self,
                 feature_generator_dict: dict,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None
                 ):

        super().__init__(feature_generator_dict, list_of_dataset, launches, metrics_name, fedot_params)
        self.aggregator = AggregationFeatures()
        self.wavelet_extractor = WaveletExtractor
        self.wavelet_list = feature_generator_dict['wavelet_types']
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.logger = Logger().get_logger()

    def _ts_chunk_function(self, ts):
        ts = self.check_Nan(ts)

        threshold_range = [1, 3, 5, 7, 9]

        spectr = self.wavelet_extractor(time_series=ts, wavelet_name=self.wavelet)
        high_freq, low_freq = spectr.decompose_signal()

        hf_lambda_peaks = lambda x: len(spectr.detect_peaks(high_freq, mph=x + 1))
        hf_lambda_names = lambda x: 'HF_peaks_higher_than_{}'.format(x + 1)
        hf_lambda_KNN = lambda x: len(spectr.detect_peaks(high_freq, mpd=x))
        hf_lambda_KNN_names = lambda x: 'HF_nearest_peaks_at_distance_{}'.format(x)

        LF_lambda_peaks = lambda x: len(spectr.detect_peaks(high_freq, mph=x + 1, valley=True))
        LF_lambda_names = lambda x: 'LF_peaks_higher_than_{}'.format(x + 1)
        LF_lambda_KNN = lambda x: len(spectr.detect_peaks(high_freq, mpd=x))
        LF_lambda_KNN_names = lambda x: 'LF_nearest_peaks_at_distance_{}'.format(x)

        lambda_list = [
            hf_lambda_KNN,
            LF_lambda_peaks,
            LF_lambda_KNN]

        lambda_list_names = [
            hf_lambda_KNN_names,
            LF_lambda_names,
            LF_lambda_KNN_names]

        features = list(map(hf_lambda_peaks, threshold_range))
        features_names = list(map(hf_lambda_names, threshold_range))
        for lambda_method, lambda_name in zip(lambda_list, lambda_list_names):
            features.extend(list(map(lambda_method, threshold_range)))
            features_names.extend(list(map(lambda_name, threshold_range)))

        self.count += 1
        feature_df = pd.DataFrame(data=features)
        feature_df = feature_df.T
        feature_df.columns = features_names
        return feature_df

    def generate_vector_from_ts(self, ts_frame):
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]

        components_and_vectors = list()
        s = 'Feature generation. Time series processed:'
        with tqdm(total=ts_frame.shape[0],
                  desc=self.logger.info(s),
                  unit='ts', initial=0) as pbar:
            for ts in ts_frame.values:
                components_and_vectors.append(self._ts_chunk_function(ts))
                pbar.update(1)

        self.logger.info(f'Time spent on wavelet extraction - {round((timeit.default_timer() - start), 2)} sec')
        return components_and_vectors

    def generate_features_from_ts(self, ts_frame, window_length=None):
        pass

    def extract_features(self, ts_data, dataset_name):

        (_, _), (y_train, _) = read_tsv(dataset_name)

        if self.train_feats is None:
            train_feats = self._choose_best_wavelet(ts_data, y_train)
            self.train_feats = train_feats
            return train_feats
        else:
            if self.test_feats is None:
                test_feats = self.generate_vector_from_ts(ts_data)
                test_feats = pd.concat(test_feats)
                test_feats.index = list(range(len(test_feats)))
                self.test_feats = delete_col_by_var(test_feats)
            return self.test_feats

    def _choose_best_wavelet(self, X_train, y_train):

        metric_list = []
        feature_list = []

        for wavelet in self.wavelet_list:
            self.logger.info(f'Generate features wavelet - {wavelet}')
            self.wavelet = wavelet

            train_feats = self.generate_vector_from_ts(X_train)
            train_feats = pd.concat(train_feats)

            self.logger.info(f'Validate model for wavelet  - {wavelet}')

            score_f1, score_roc_auc = self._validate_window_length(features=train_feats, target=y_train)

            self.logger.info(f'Obtained metric for wavelet {wavelet}  - F1, ROC_AUC - {score_f1, score_roc_auc}')

            metric_list.append((score_f1, score_roc_auc))
            feature_list.append(train_feats)
            self.count = 0

        max_score = [sum(x) for x in metric_list]
        index_of_window = int(max_score.index(max(max_score)))

        train_feats = feature_list[index_of_window]
        train_feats.index = list(range(len(train_feats)))

        self.wavelet = self.wavelet_list[index_of_window]
        self.logger.info(f'<{self.wavelet}> wavelet was chosen')

        train_feats = delete_col_by_var(train_feats)

        return train_feats
