from itertools import repeat
from multiprocessing import Pool
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from tqdm import tqdm

from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.signal.WindowedFeaturesExtractor import WindowedFeatureExtractor
from fedot_ind.core.operation.transformation.extraction.statistical import StatFeaturesExtractor


class SignalExtractor(WindowedFeatureExtractor):
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

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.ts_samples_count = None
        self.aggregator = StatFeaturesExtractor()
        self.wavelet_extractor = WaveletExtractor

        self.wavelet = params.get('wavelet')
        self.vis_flag = False
        self.train_feats = None
        self.test_feats = None
        self.dict_of_methods = {'Peaks': self._method_of_peaks,
                                'AC': self._method_of_AC}

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

    def _method_of_AC(self, specter, level: int = 3):
        high_freq, low_freq = specter.decompose_signal()
        hf_AC_features = specter.generate_features_from_AC(HF=high_freq,
                                                           LF=low_freq,
                                                           level=level)

        feature_df = pd.concat(hf_AC_features, axis=1)
        return feature_df

    def _ts_chunk_function(self, ts,
                           method_name: str = 'AC'):

        ts = self.check_for_nan(ts)
        specter = self.wavelet_extractor(time_series=ts, wavelet_name=self.wavelet)
        feature_df = self.dict_of_methods[method_name](specter)

        return feature_df

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
                                                         zip(ts_frame, repeat(method_name))),
                                               total=ts_samples_count,
                                               desc='Feature Generation. TS processed',
                                               unit=' ts',
                                               colour='black'
                                               )
                                          )

        return components_and_vectors

    def generate_features_from_ts(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
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
