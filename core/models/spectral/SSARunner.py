import os
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from core.metrics.metrics_implementation import ParetoMetrics
from core.models.ExperimentRunner import ExperimentRunner
from core.models.spectral.spectrum_decomposer import SpectrumDecomposer
from core.models.statistical.stat_features_extractor import StatFeaturesExtractor
from core.operation.utils.Decorators import time_it
from core.operation.utils.utils import PROJECT_PATH


class SSARunner(ExperimentRunner):
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

    def __init__(self, window_sizes: dict,
                 window_mode: bool = False,
                 use_cache: bool = False):

        super().__init__()
        self.use_cache = use_cache
        self.aggregator = StatFeaturesExtractor()
        self.spectrum_extractor = SpectrumDecomposer
        self.pareto_front = ParetoMetrics()
        self.window_sizes = window_sizes
        self.vis_flag = False
        self.rank_hyper = None
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    @staticmethod
    def __vis_and_save_components(components_df):

        n_rows = round(components_df[0].shape[1] / 5)

        if n_rows < 4:
            plot_area = 'small'
        elif 5 < n_rows < 9:
            plot_area = 'mid'
        else:
            plot_area = 'big'

        plot_dict = {'small': (20, 10),
                     'mid': (20, 10),
                     'big': (40, 20)}

        fig_size = plot_dict[plot_area]

        for idx, df in enumerate(components_df):
            df.plot(subplots=True,
                    figsize=fig_size,
                    legend=None,
                    layout=(n_rows + 1, 5))

            plt.tight_layout()

            path_to_save = os.path.join(PROJECT_PATH, 'results_of_experiments', 'components', f'components_{idx}.png')
            plt.savefig(path_to_save)

    def _ts_chunk_function(self, ts_data: pd.DataFrame) -> list:

        ts = self.check_for_nan(ts_data)
        specter = self.spectrum_extractor(time_series=ts, window_length=self.current_window)

        ts_comps, x_elem, v, components_df, _, n_components, explained_dispersion = specter.decompose(
            rank_hyper=self.rank_hyper)

        return [components_df, n_components, explained_dispersion]

    def generate_vector_from_ts(self, ts_frame):
        ts_samples_count = ts_frame.shape[0]

        components_and_vectors = list()
        with tqdm(total=ts_samples_count,
                  desc='Feature Generation. Samples processed: ',
                  unit=' samples', initial=0, colour='black') as pbar:
            for ts in ts_frame.values:
                v = self._ts_chunk_function(ts)
                components_and_vectors.append(v)
                pbar.update(1)
        return components_and_vectors

    @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:

        if self.current_window is None:
            self._choose_best_window_size(ts_data, dataset_name=dataset_name)
            aggregation_df = self.delete_col_by_var(self.train_feats)
        else:
            eigenvectors_and_rank = self.generate_vector_from_ts(ts_data)
            eigenvectors_list_test = [x[0].iloc[:, :self.min_rank] for x in eigenvectors_and_rank]

            aggregation_df = self.generate_features_from_ts(eigenvectors_list_test, window_mode=self.window_mode)
            aggregation_df = aggregation_df[self.train_feats.columns]

            for col in aggregation_df.columns:
                aggregation_df[col].fillna(value=aggregation_df[col].mean(), inplace=True)

        return aggregation_df

    def generate_features_from_ts(self, eigenvectors_list: list, window_mode: bool = False) -> pd.DataFrame:

        if window_mode:
            gen = self.aggregator.create_baseline_features
            lambda_function_for_stat_features = lambda x: self.apply_window_for_stat_feature(x.T,
                                                                                             feature_generator=gen)
            lambda_function_for_concat = lambda x: pd.concat(x, axis=1)

            list_with_stat_features_on_interval = list(map(lambda_function_for_stat_features, eigenvectors_list))
            aggregation_df = list(map(lambda_function_for_concat, list_with_stat_features_on_interval))
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
        metric_list = []
        n_comp_list = []
        eigen_list = []
        # TODO: check if it is possible to use multiprocessing here
        # TODO: check window sizes
        window_list = self.window_sizes[dataset_name]

        for window_length in window_list:
            self.logger.info(f'Generate features for window length - {window_length}')
            self.current_window = window_length

            eigenvectors_and_rank = self.generate_vector_from_ts(x_train.sample(frac=0.5))

            rank_list = [x[1] for x in eigenvectors_and_rank]

            explained_dispersion = [x[2] for x in eigenvectors_and_rank]
            mean_dispersion = np.mean(explained_dispersion)
            self.explained_dispersion = round(float(mean_dispersion))

            self.n_components = Counter(rank_list).most_common(n=1)[0][0]

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

        # index_of_window = int(metric_list.index(max(metric_list)))
        index_of_window = self.pareto_front.pareto_metric_list(np.array(metric_list))
        index_of_window = np.where(index_of_window == True)[0][0]
        self.current_window = window_list[index_of_window]
        eigenvectors_list = eigen_list[index_of_window]
        self.min_rank = int(np.round(np.mean([x.shape[1] for x in eigenvectors_list])))
        self.rank_hyper = self.min_rank
        eigenvectors_and_rank = self.generate_vector_from_ts(x_train)
        self.eigenvectors_list_train = [x[0].iloc[:, :self.min_rank] for x in eigenvectors_and_rank]
        self.train_feats = self.generate_features_from_ts(self.eigenvectors_list_train, window_mode=self.window_mode)
        self.train_feats = self.delete_col_by_var(self.train_feats)
        for col in self.train_feats.columns:
            self.train_feats[col].fillna(value=self.train_feats[col].mean(), inplace=True)

        self.n_components = n_comp_list[index_of_window]
        self.logger.info(f'Window length = {self.current_window} was chosen')

        return self.train_feats

    @staticmethod
    def threading_operation(ts_frame: pd.DataFrame,
                            function_for_feature_extraction: callable):
        pool = Pool(8)
        features = pool.map(function_for_feature_extraction, ts_frame)
        pool.close()
        pool.join()
        return features
