from multiprocessing import Pool
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fedot.core.operations.operation_parameters import OperationParameters
from tqdm import tqdm

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.metrics.metrics_implementation import ParetoMetrics
from fedot_ind.core.models.BaseExtractor import BaseExtractor
from fedot_ind.core.operation.decomposition.SpectrumDecomposition import SpectrumDecomposer
from fedot_ind.core.operation.transformation.data.eigen import combine_eigenvectors
from fedot_ind.core.operation.transformation.extraction.statistical import StatFeaturesExtractor
from fedot_ind.core.operation.transformation.regularization.spectrum import sv_to_explained_variance_ratio


class SSAExtractor(BaseExtractor):
    """Class responsible for spectral feature generator experiment.

    Args:
        window_sizes: dict of window sizes for SSA algorithm.
        window_mode: flag for window mode. If True, SSA algorithm will be applied to each window of time series.
        use_cache: flag for cache usage. If True, SSA algorithm will be applied to each window of time series.

    Attributes:
        aggregator (StatFeaturesExtractor): class for statistical features extraction.
        spectrum_extractor (SpectrumDecomposer): class for SSA algorithm.
        pareto_front (ParetoMetrics): class for pareto front calculation.
        vis_flag (bool): flag for visualization.
        rank_hyper (int): ...
        train_feats (pd.DataFrame): extracted features for train data.
        test_feats (pd.DataFrame): extracted features for test data.

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.combine_eigenvectors = params.get('combine_eigenvectors')
        self.correlation_level = params.get('correlation_level')
        self.use_cache = params.get('use_cache')
        self.window_sizes = params.get('window_sizes')
        self.window_mode = params.get('window_mode')
        self.spectral_hyperparams = params.get('spectral_hyperparams')
        self.__init_spectral_hyperparams()

        self.aggregator = StatFeaturesExtractor()
        self.spectrum_extractor = SpectrumDecomposer
        self.pareto_front = ParetoMetrics()
        self.datacheck = DataCheck()

        self.vis_flag = False
        self.rank_hyper = None
        self.train_feats = None
        self.test_feats = None
        self.n_components = None

    def __init_spectral_hyperparams(self):
        if self.spectral_hyperparams is not None:
            for k, v in self.spectral_hyperparams.items():
                setattr(self, k, v)

    @staticmethod
    def visualise(eigenvectors_list):

        n_rows = max((1, round(eigenvectors_list[0].shape[1] / 5)))

        if n_rows < 4:
            plot_area = 'small'
        elif 5 < n_rows < 9:
            plot_area = 'mid'
        else:
            plot_area = 'big'

        plot_dict = {'small': (15, 5),
                     'mid': (30, 10),
                     'big': (40, 20)}

        fig_size = plot_dict[plot_area]

        for idx, df in enumerate(eigenvectors_list):
            df.plot(subplots=True,
                    figsize=fig_size,
                    legend=None,
                    layout=(n_rows, df.shape[1]))

            plt.tight_layout()
            plt.show()

    def _ts_chunk_function(self, ts_data: pd.DataFrame) -> list:

        ts = self.check_for_nan(ts_data)
        specter = self.spectrum_extractor(time_series=ts, window_length=self.current_window)

        TS_comps, Sigma, rank, X_elem, V = specter.decompose(
            rank_hyper=self.rank_hyper)
        explained_variance, n_components = sv_to_explained_variance_ratio(Sigma, rank)

        if self.combine_eigenvectors:
            components_df = combine_eigenvectors(TS_comps, rank, self.correlation_level)
        else:
            components_df = specter.components_to_df(TS_comps, rank)

        return [components_df, n_components, explained_variance]

    def generate_vector_from_ts(self, ts_frame):
        ts_samples_count = ts_frame.shape[0]
        n_processes = self.n_processes
        self.logger.info(f'Number of processes: {n_processes}')

        with Pool(n_processes) as p:
            components_and_vectors = list(tqdm(p.imap(self._ts_chunk_function,
                                                      ts_frame.values),
                                               total=ts_samples_count,
                                               desc='Feature Generation. TS processed',
                                               unit=' ts',
                                               colour='black'
                                               )
                                          )
        self.logger.info(f'Number of time series processed: {ts_samples_count}')
        return components_and_vectors

    # @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None, target: np.ndarray = None) -> pd.DataFrame:
        self.logger.info('Spectral features extraction started')
        if self.current_window is None:
            self._choose_best_window_size(ts_data, dataset_name=dataset_name)
            aggregation_df = self.train_feats
        else:
            eigenvectors_list_test = ts_data
            self.eigenvectors_list_test = self.check_rank_consistency(eigenvectors_list_test)
            aggregation_df = self.generate_features_from_ts(eigenvectors_list_test, window_mode=self.window_mode)
            aggregation_df = aggregation_df[self.train_feats.columns]

            for col in aggregation_df.columns:
                aggregation_df[col].fillna(value=aggregation_df[col].mean(), inplace=True)

        self.logger.info('Spectral features extraction finished')
        return aggregation_df

    def generate_features_from_ts(self, eigenvectors_list: list, window_mode: bool = False) -> pd.DataFrame:
        eigenvectors_list = list(map(lambda x: self.datacheck.check_data(x, return_df=True), eigenvectors_list))
        if self.window_mode:
            gen = self.aggregator.create_baseline_features
            function_for_stat_features = lambda x: self.apply_window_for_stat_feature(ts_data=x.T,
                                                                                      feature_generator=gen,
                                                                                      window_size=self.current_window)
            concat_function = lambda x: pd.concat(x, axis=1)

            list_with_stat_features_on_interval = list(map(function_for_stat_features, eigenvectors_list))
            aggregation_df = list(map(concat_function, list_with_stat_features_on_interval))
        else:
            aggregation_df = list(map(lambda x: self.aggregator.create_baseline_features(x.T), eigenvectors_list))

        components_names = aggregation_df[0].index.values
        columns_names = aggregation_df[0].columns.values

        aggregation_df = pd.concat([pd.DataFrame(x.values.ravel()) for x in aggregation_df], axis=1)
        aggregation_df.columns = range(len(aggregation_df.columns))
        aggregation_df = aggregation_df.T

        new_column_names = []
        for number_of_component in components_names:
            new_column_names.extend([f'{x}_for_component: {number_of_component}' for x in columns_names])

        aggregation_df.columns = new_column_names

        return aggregation_df

    def _choose_best_window_size(self, x_train: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Chooses the best window for feature extraction.

        Args:
            x_train: train data.
            dataset_name: name of dataset.

        Returns:
            Extracted features for train data.

        """
        self.logger.info('Choose best window size for feature extraction')
        metric_list = []
        n_comp_list = []
        eigen_list = []

        if self.window_sizes == 'auto':
            ts_length = x_train.shape[1]
            window_list = list(map(lambda x: round(ts_length / x), [10, 5, 3]))
        else:
            window_list = self.window_sizes

        for window_length in window_list:
            self.logger.info(f'Generate features for window length - {window_length}')
            self.current_window = window_length

            eigenvectors_and_rank = self.generate_vector_from_ts(x_train.sample(frac=0.5))

            rank_list = [x[1] for x in eigenvectors_and_rank]

            explained_dispersion = [x[2] for x in eigenvectors_and_rank]
            median_dispersion = np.median(explained_dispersion)
            self.explained_dispersion = round(float(median_dispersion))

            self.n_components = int(np.median(rank_list))

            eigenvectors_list = [x[0].iloc[:, :self.n_components] for x in eigenvectors_and_rank]

            self.logger.info(f'Every eigenvector with impact less then 1 % percent was eliminated. '
                             f'{self.explained_dispersion} % of explained dispersion '
                             f'obtained by first - {self.n_components} components.')
            signal_compression = 100 * (1 - (self.n_components / window_length))
            explained_dispersion = self.explained_dispersion
            metrics = [signal_compression, explained_dispersion]
            metric_list.append(metrics)

            eigen_list.append(eigenvectors_list)
            n_comp_list.append(self.n_components)
            if self.n_components > 15:
                self.logger.info(f'SSA method find {self.n_components} PCT.'
                                 f'This is mean that SSA method does not find effective low rank structure for '
                                 f'this signal.')
                break

        index_of_window = self.pareto_front.pareto_metric_list(np.array(metric_list))
        index_of_window = np.where(index_of_window == True)[0][0]
        self.current_window = window_list[index_of_window]
        self.logger.info(f'Best window length - {self.current_window} selected.')

        eigenvectors_list = eigen_list[index_of_window]
        self.rank_hyper = int(np.round(np.median([x.shape[1] for x in eigenvectors_list])))
        eigenvectors_and_rank = self.generate_vector_from_ts(x_train)
        eigenvectors_list_train = [x[0].iloc[:, :self.rank_hyper] for x in eigenvectors_and_rank]
        self.eigenvectors_list_train = self.check_rank_consistency(eigenvectors_list_train)
        self.train_feats = self.generate_features_from_ts(self.eigenvectors_list_train, window_mode=self.window_mode)
        for col in self.train_feats.columns:
            self.train_feats[col].fillna(value=self.train_feats[col].mean(), inplace=True)
        self.train_feats = self.delete_col_by_var(self.train_feats)
        self.n_components = n_comp_list[index_of_window]
        self.logger.info(f'Window length = {self.current_window} was chosen')

        return self.train_feats

    def __create_mask(self, eigenvectors_list_train):
        mask = [x.shape[1] < self.rank_hyper for x in eigenvectors_list_train]
        invalid_idx = [i for i, x in enumerate(mask) if x]
        return invalid_idx

    def check_rank_consistency(self, eigenvectors_list, rank=None):
        if rank is None:
            rank = self.rank_hyper

        invalid_idx = self.__create_mask(eigenvectors_list)
        while len(invalid_idx) != 0:
            eigenvectors_list = self.__fill_empty_eigenvectors(invalid_idx, eigenvectors_list, rank)
            invalid_idx = self.__create_mask(eigenvectors_list)
            if len(invalid_idx) == 0:
                break
        return eigenvectors_list

    def __fill_empty_eigenvectors(self, invalid_idx, eigenvectors_list, rank):
        for idx in invalid_idx:
            invalid_sample = eigenvectors_list[idx]
            missed_col = rank - invalid_sample.shape[1]
            last_number_of_component = invalid_sample.columns.values[-1]
            if missed_col == 1:
                eigenvectors_list[idx][last_number_of_component + missed_col] = 0
            else:
                for number_of_missed_component in range(1, missed_col):
                    eigenvectors_list[idx][last_number_of_component + number_of_missed_component] = 0

        return eigenvectors_list
