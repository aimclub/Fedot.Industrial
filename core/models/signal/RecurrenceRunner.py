import timeit
from tqdm import tqdm

from core.models.ExperimentRunner import ExperimentRunner
from core.metrics.metrics_implementation import *
from core.operation.transformation.TS import TSTransformer
from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import read_tsv


class RecurrenceRunner(ExperimentRunner):
    """
    Class responsible for wavelet feature generator experiment.

    :wavelet_types: list of wavelet types to be used in experiment. Defined in Config_Classification.yaml
    """

    def __init__(self, window_mode: bool = False):

        super().__init__()

        self.ts_samples_count = None
        self.window_mode = window_mode
        self.transformer = TSTransformer
        self.train_feats = None
        self.test_feats = None
        self.logger = Logger().get_logger()

    def _ts_chunk_function(self,
                           ts):

        ts = self.check_for_nan(ts)
        specter = self.transformer(time_series=ts)
        feature_df = pd.Series(specter.get_reccurancy_metrics())
        return feature_df

    def generate_vector_from_ts(self, ts_frame: pd.DataFrame) -> list:
        """
        Generate vector from time series.

        :param ts_frame: time series dataframe
        :return:
        """
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]

        components_and_vectors = list()
        with tqdm(total=ts_frame.shape[0],
                  desc='Feature generation. Time series processed:',
                  unit='ts', initial=0) as pbar:
            for ts in ts_frame.values:
                components_and_vectors.append(self._ts_chunk_function(ts))
                pbar.update(1)
        self.logger.info('Feature generation finished. TS processed: {}'.format(ts_frame.shape[0]))
        components_and_vectors = pd.concat(components_and_vectors, axis=1).T
        time_elapsed = round(timeit.default_timer() - start, 2)
        self.logger.info(f'Time spent on recurrence features extraction - {time_elapsed} sec')
        return components_and_vectors

    def extract_features(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        self.logger.info('Recurrence feature extraction started')

        (_, _), (y_train, _) = read_tsv(dataset_name)

        if self.train_feats is None:
            train_feats = self.generate_vector_from_ts(ts_data)
            self.train_feats = train_feats
            return self.train_feats
        else:
            test_feats = self.generate_vector_from_ts(ts_data)
            self.test_feats = test_feats[self.train_feats.columns]
            return self.test_feats
