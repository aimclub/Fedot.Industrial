from itertools import repeat
from multiprocessing import Pool
from typing import Optional
from tqdm import tqdm

from core.metrics.metrics_implementation import *
from core.models.BaseExtractor import BaseExtractor
from core.operation.transformation.basis.wavelet import WaveletBasisImplementation
from core.operation.transformation.extraction.statistical import StatFeaturesExtractor
from fedot.core.operations.operation_parameters import OperationParameters


class SignalExtractor(BaseExtractor):
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
        self.aggregator = StatFeaturesExtractor()
        self.n_components = params.get('n_components', 2)
        self.wavelet = params.get('wavelet')
        self.wavelet_basis = WaveletBasisImplementation(params)
        self.feature_generator = StatFeaturesExtractor()

    def _ts_chunk_function(self, ts):

        ts = self.check_for_nan(ts)
        high_freq, low_freq = self.wavelet_basis._transform(series=ts)
        hf_AC_features = self.generate_features_from_AC(HF=high_freq,
                                                        LF=low_freq)
        feature_df = pd.concat(hf_AC_features, axis=1)
        return feature_df

    def generate_features_from_AC(self,
                                  HF,
                                  LF):

        feature_list = []
        feature_df = self.feature_generator.create_features(LF)
        feature_df.columns = [f'{x}_LF' for x in feature_df.columns]
        feature_list.append(feature_df)
        if self.wavelet in self.wavelet_basis.continuous_wavelets:
            for idx, level in enumerate(HF):
                feature_df = self.feature_generator.create_features(level)
                feature_df.columns = [f'{x}_level_{idx}' for x in feature_df.columns]
                feature_list.append(feature_df)
        else:
            for i in range(self.n_components):
                feature_df = self.feature_generator.create_features(HF)
                feature_df.columns = [f'{x}_level_{i}' for x in feature_df.columns]
                feature_list.append(feature_df)
                HF = self.wavelet_basis._decompose_signal(input_data=HF)[0]
        return feature_list

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

        self.logger.info('Feature generation finished. TS processed: {}'.format(ts_frame.shape[0]))
        return components_and_vectors

    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        self.logger.info('Wavelet feature generation started')

        ts_data = np.array(ts_data)
        if len(ts_data.shape) == 1:
            self.test_feats = self._ts_chunk_function(ts_data)
        else:
            test_feats = self.generate_vector_from_ts(ts_data)
            # test_feats = self.generate_vector_from_ts(np.array(ts_data))
            test_feats = pd.concat(test_feats)
            test_feats.index = list(range(len(test_feats)))
            self.test_feats = test_feats

        self.logger.info('Wavelet feature generation finished')
        return self.test_feats
