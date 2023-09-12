from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task

from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.WindowedFeaturesExtractor import WindowedFeatureExtractor
from fedot_ind.core.models.quantile.quantile_extractor import QuantileExtractor
from fedot_ind.core.operation.transformation.basis.wavelet import WaveletBasisImplementation


class SignalExtractor(WindowedFeatureExtractor):
    """Class responsible for wavelet feature generator experiment.
    Args:
        wavelet_types: list of wavelet types to be used in experiment. Defined in Config_Classification.yaml.
        use_cache: flag to use cache or not. Defined in Config_Classification.yaml
    Attributes:
        ts_samples_count (int): number of samples in time series
        # aggregator (StatFeaturesExtractor): class to aggregate features
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
        self.aggregator = QuantileExtractor({'window_mode': False,
                                             'window_size': 10,
                                             'var_threshold': 0})
        self.wavelet_basis = WaveletBasisImplementation
        self.n_components = params.get('n_components')
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

    def generate_features_from_ts(self, ts_frame: pd.DataFrame, dataset_name: str = None) -> list:
        """Generate vector from time series.
        Args:
            ts_frame (pd.DataFrame): time series to be transformed.
            method_name (str): method to be used for transformation.
        Returns:
            list: list of components and vectors.
        """

        wavelet_basis = self.wavelet_basis({'n_components': self.n_components,
                                            'wavelet': self.wavelet})
        transformed_ts = wavelet_basis._transform(ts_frame)
        input_transformed_ts = InputData(idx=np.arange(len(transformed_ts)),
                                         features=transformed_ts,
                                         target=transformed_ts,
                                         task=Task(TaskTypesEnum.regression),
                                         data_type=DataTypesEnum.image)

        extracted_features_train = self.aggregator.transform(input_transformed_ts)
        return extracted_features_train.predict
