from multiprocessing import Pool
from typing import Optional
from sklearn.preprocessing import StandardScaler
from fedot.core.operations.operation_parameters import OperationParameters
from joblib import Parallel, delayed
from tqdm import tqdm
from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.core.metrics.metrics_implementation import *
from fedot_ind.core.models.WindowedFeaturesExtractor import WindowedFeatureExtractor
from fedot_ind.core.operation.transformation.DataTransformer import TSTransformer
from fedot_ind.core.models.recurrence.sequences import ReccurenceFeaturesExtractor


class RecurrenceExtractor(WindowedFeatureExtractor):
    """Class responsible for wavelet feature generator experiment.
    Args:
        window_mode: boolean flag - if True, window mode is used. Defaults to False.
        use_cache: boolean flag - if True, cache is used. Defaults to False.
    Attributes:
        transformer: TSTransformer object.
        self.extractor: ReccurenceExtractor object.
        train_feats: train features.
        test_feats: test features.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.image_mode = False

        self.window_mode = params.get('window_mode')
        self.min_signal_ratio = params.get('min_signal_ratio')
        self.max_signal_ratio = params.get('max_signal_ratio')
        self.rec_metric = params.get('rec_metric')
        self.window_size = 10
        self.transformer = TSTransformer
        self.extractor = ReccurenceFeaturesExtractor
        self.train_feats = None
        self.test_feats = None

    def _ts_chunk_function(self, ts):

        ts = StandardScaler().fit_transform(ts.reshape(-1, 1))
        ts = ts.reshape(-1)

        if self.window_mode:
            trajectory_transformer = HankelMatrix(time_series=ts, window_size=self.window_size)
            ts = trajectory_transformer.trajectory_matrix
            self.ts_length = trajectory_transformer.ts_length

        specter = self.transformer(time_series=ts,
                                   min_signal_ratio=self.min_signal_ratio,
                                   max_signal_ratio=self.max_signal_ratio,
                                   rec_metric=self.rec_metric)
        feature_df = specter.ts_to_recurrence_matrix()
        if not self.image_mode:
            feature_df = pd.Series(self.extractor(recurrence_matrix=feature_df).recurrence_quantification_analysis())
        feature_df = feature_df.fillna(0)
        return feature_df

    def generate_vector_from_ts(self, ts_frame: pd.DataFrame) -> pd.DataFrame:
        """Generate vector from time series.
        Args:
            ts_frame: time series frame
        Returns:
            Feature vector
        """

        parallel = Parallel(n_jobs=self.n_processes, verbose=0, pre_dispatch="2*n_jobs")
        components_and_vectors = parallel(delayed(self.generate_features_from_ts)(sample) for sample in ts_frame.value)
        if self.image_mode:
            components_and_vectors = np.asarray(components_and_vectors)
            components_and_vectors = components_and_vectors[:, np.newaxis, :, :]
        else:
            components_and_vectors = pd.concat(components_and_vectors, axis=1).T
        return components_and_vectors

    def generate_features_from_ts(self, ts_data: pd.DataFrame,
                                  window_length: int = None) -> pd.DataFrame:

        if self.train_feats is None:
            train_feats = self.generate_vector_from_ts(ts_data)
            self.train_feats = train_feats
            return self.train_feats
        else:
            test_feats = self.generate_vector_from_ts(ts_data)
            if self.image_mode:
                self.test_feats = test_feats
            else:
                self.test_feats = test_feats[self.train_feats.columns]
            return self.test_feats