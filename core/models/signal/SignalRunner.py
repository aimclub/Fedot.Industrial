from itertools import repeat
from multiprocessing import Pool

from tqdm import tqdm

from core.metrics.metrics_implementation import *
from core.models.ExperimentRunner import ExperimentRunner
from core.operation.transformation.extraction.wavelet import WaveletExtractor
from core.operation.transformation.extraction.statistical import StatFeaturesExtractor
from core.architecture.abstraction.Decorators import time_it, dataframe_adapter


class SignalRunner(ExperimentRunner):
    """Class responsible for wavelet feature generator experiment.

    Args:
        wavelet_types: list of wavelet types to be used in experiment. Defined in Config_Classification.yaml.
        use_cache: flag to use cache or not. Defined in Config_Classification.yaml

    Attributes:
        ts_samples_count (int): number of samples in time series
        aggregator (StatFeaturesExtractor): class to aggregate features
        wavelet_extractor (WaveletExtractor): class to extract wavelet features
        wavelet (str): current wavelet type
        vis_flag (bool): flag to visualize or not
        train_feats (pd.DataFrame): train features
        test_feats (pd.DataFrame): test features
        dict_of_methods (dict): dictionary of methods to extract features

    """

    def __init__(self, wavelet_types: list = ('db5', 'sym5', 'coif5', 'bior2.4'),
                 use_cache: bool = False):

        super().__init__()
        self.use_cache = use_cache
        self.ts_samples_count = None
        self.aggregator = StatFeaturesExtractor()
        self.wavelet_extractor = WaveletExtractor
        self.wavelet_list = wavelet_types
        self.wavelet = None
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.dict_of_methods = {'Peaks': self._method_of_peaks,
                                'AC': self._method_of_ac}

    def _method_of_peaks(self, specter):
        threshold_range = [1, 3, 5, 7, 9]
        high_freq, low_freq = specter.decompose_signal()

        hf_lambda_peaks = lambda x: len(specter.detect_peaks(high_freq, mph=x + 1))
        hf_lambda_names = lambda x: 'HF_peaks_higher_than_{}'.format(x + 1)
        hf_lambda_knn = lambda x: len(specter.detect_peaks(high_freq, mpd=x))
        hf_lambda_knn_names = lambda x: 'HF_nearest_peaks_at_distance_{}'.format(x)

        lf_lambda_peaks = lambda x: len(specter.detect_peaks(high_freq, mph=x + 1, valley=True))
        lf_lambda_names = lambda x: 'LF_peaks_higher_than_{}'.format(x + 1)
        lf_lambda_knn = lambda x: len(specter.detect_peaks(high_freq, mpd=x))
        lf_lambda_knn_names = lambda x: 'LF_nearest_peaks_at_distance_{}'.format(x)

        lambda_list = [
            hf_lambda_knn,
            lf_lambda_peaks,
            lf_lambda_knn]

        lambda_list_names = [
            hf_lambda_knn_names,
            lf_lambda_names,
            lf_lambda_knn_names]

        features = list(map(hf_lambda_peaks, threshold_range))
        features_names = list(map(hf_lambda_names, threshold_range))
        for lambda_method, lambda_name in zip(lambda_list, lambda_list_names):
            features.extend(list(map(lambda_method, threshold_range)))
            features_names.extend(list(map(lambda_name, threshold_range)))

        feature_df = pd.DataFrame(data=features)
        feature_df = feature_df.T
        feature_df.columns = features_names
        return feature_df

    def _method_of_ac(self, specter, level: int = 3):
        high_freq, low_freq = specter.decompose_signal()
        hf_ac_features = specter.generate_features_from_AC(HF=high_freq,
                                                           LF=low_freq,
                                                           level=level)

        feature_df = pd.concat(hf_ac_features, axis=1)
        return feature_df

    def _ts_chunk_function(self, ts,
                           method_name: str = 'AC'):

        ts = self.check_for_nan(ts)
        specter = self.wavelet_extractor(time_series=ts, wavelet_name=self.wavelet)
        feature_df = self.dict_of_methods[method_name](specter)

        return feature_df

    @dataframe_adapter
    def generate_vector_from_ts(self, ts_frame: pd.DataFrame, method_name: str = 'AC') -> list:
        """Generate vector from time series.

        Args:
            ts_frame (pd.DataFrame): time series to be transformed.
            method_name (str): method to be used for transformation.

        Returns:
            list: list of components and vectors.

        """
        ts_samples_count = ts_frame.shape[0]
        n_processes = self.n_processes
        with Pool(n_processes) as p:
            components_and_vectors = list(tqdm(p.starmap(self._ts_chunk_function,
                                                         zip(ts_frame.values, repeat(method_name))),
                                               total=ts_samples_count,
                                               desc='Feature Generation. TS processed',
                                               unit=' ts',
                                               colour='black'
                                               )
                                          )

        self.logger.info('Feature generation finished. TS processed: {}'.format(ts_frame.shape[0]))
        return components_and_vectors

    @time_it
    def get_features(self, ts_data: pd.DataFrame,
                     dataset_name: str = None) -> pd.DataFrame:

        if not self.wavelet:
            train_feats = self._choose_best_wavelet(ts_data)
            self.train_feats = train_feats
            return self.train_feats
        else:
            test_feats = self.generate_vector_from_ts(ts_data)
            test_feats = pd.concat(test_feats)
            test_feats.index = list(range(len(test_feats)))
            self.test_feats = test_feats
        return self.test_feats

    def _choose_best_wavelet(self, x_train: pd.DataFrame) -> pd.DataFrame:
        """Chooses the best wavelet for feature extraction.

        Args:
            x_train: train features.

        Returns:
            pd.DataFrame: features with the best wavelet.
        """
        metric_list = []
        feature_list = []

        for wavelet in self.wavelet_list:
            self.logger.info(f'Generate features wavelet - {wavelet}')
            self.wavelet = wavelet

            train_feats = self.generate_vector_from_ts(x_train)
            train_feats = pd.concat(train_feats)
            filtered_df = self.delete_col_by_var(train_feats)
            feature_list.append(train_feats)
            metric_list.append((filtered_df.shape[0], filtered_df.shape[1]))

        max_score = [sum(x) for x in metric_list]
        index_of_window = int(max_score.index(max(max_score)))

        train_feats = feature_list[index_of_window]
        train_feats.index = list(range(len(train_feats)))

        self.wavelet = self.wavelet_list[index_of_window]
        self.logger.info(f'<{self.wavelet}> wavelet was chosen')

        train_feats = self.delete_col_by_var(train_feats)
        return train_feats
