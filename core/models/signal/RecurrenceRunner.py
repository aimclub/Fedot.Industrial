from tqdm import tqdm

from core.metrics.metrics_implementation import *
from core.models.ExperimentRunner import ExperimentRunner
from core.operation.transformation.TS import TSTransformer
from core.operation.utils.Decorators import time_it


class RecurrenceRunner(ExperimentRunner):
    """Class responsible for wavelet feature generator experiment.

    Args:
        window_mode (bool): boolean flag - if True, window mode is used. Defaults to False.
        use_cache (bool): boolean flag - if True, cache is used. Defaults to False.
    """

    def __init__(self, window_mode: bool = False, use_cache: bool = False):

        super().__init__()

        self.ts_samples_count = None
        self.window_mode = window_mode
        self.use_cache = use_cache
        self.transformer = TSTransformer
        self.train_feats = None
        self.test_feats = None

    def _ts_chunk_function(self, ts):

        ts = self.check_for_nan(ts)
        specter = self.transformer(time_series=ts)
        feature_df = pd.Series(specter.get_reccurancy_metrics())
        return feature_df

    def generate_vector_from_ts(self, ts_frame: pd.DataFrame) -> pd.DataFrame:
        """Generate vector from time series.

        Args:
            ts_frame (pd.DataFrame): time series frame

        Returns:
            pd.DataFrame: feature vector
        """
        self.ts_samples_count = ts_frame.shape[0]

        components_and_vectors = list()
        with tqdm(total=ts_frame.shape[0],
                  desc='Feature generation. Time series processed:',
                  unit='ts', initial=0, colour='black') as pbar:
            for ts in ts_frame.values:
                components_and_vectors.append(self._ts_chunk_function(ts))
                pbar.update(1)
        components_and_vectors = pd.concat(components_and_vectors, axis=1).T

        return components_and_vectors

    @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        self.logger.info('Recurrence feature extraction started')

        if self.train_feats is None:
            train_feats = self.generate_vector_from_ts(ts_data)
            self.train_feats = train_feats
            return self.train_feats
        else:
            test_feats = self.generate_vector_from_ts(ts_data)
            self.test_feats = test_feats[self.train_feats.columns]
            return self.test_feats
