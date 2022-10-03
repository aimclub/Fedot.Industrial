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


class SSARunner(ExperimentRunner):
    """
    Class responsible for spectral feature generator experiment
        :param window_sizes: list of window sizes to be used for feature extraction
        :param window_mode: boolean flag - if True, window mode is used
    """

    def __init__(self, window_sizes: list,
                 window_mode: bool = False,
                 use_cache: bool = False):

        super().__init__()
        self.use_cache = use_cache
        self.ts_samples_count = None
        self.aggregator = StatFeaturesExtractor()
        self.spectrum_extractor = SpectrumDecomposer
        self.pareto_front = ParetoMetrics()
        self.window_length_list = window_sizes

        if isinstance(self.window_length_list, int):
            self.window_length_list = [self.window_length_list]

        self.vis_flag = False
        self.rank_hyper = None
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def __vis_and_save_components(self, components_df):

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

        figsize = plot_dict[plot_area]
        layout = (n_rows + 1, 5)

        for idx, df in enumerate(components_df):
            df.plot(subplots=True,
                    figsize=figsize,
                    legend=None,
                    layout=(n_rows + 1, 5))

            plt.tight_layout()
            plt.savefig(os.path.join(self.path_to_save_png, f'components_for_ts_class_{idx}.png'))

    def _ts_chunk_function(self, ts_data: pd.DataFrame) -> list:

        ts = self.check_for_nan(ts_data)
        specter = self.spectrum_extractor(time_series=ts, window_length=self.window_length)

        ts_comps, x_elem, v, components_df, _, n_components, explained_dispersion = specter.decompose(
            rank_hyper=self.rank_hyper)

        return [components_df, n_components, explained_dispersion]

    def generate_vector_from_ts(self, ts_frame):
        self.ts_samples_count = ts_frame.shape[0]

        components_and_vectors = list()
        with tqdm(total=self.ts_samples_count,
                  desc='Feature Generation. Samples processed: ',
                  unit=' samples', initial=0) as pbar:
            for ts in ts_frame.values:
                v = self._ts_chunk_function(ts)
                components_and_vectors.append(v)
                pbar.update(1)
        return components_and_vectors

    @time_it
    def get_features(self, ts_data: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:

        if self.window_length is None:
            self._choose_best_window_size(ts_data, dataset_name=dataset_name)
            aggregation_df = self.delete_col_by_var(self.train_feats)
        else:
            eigenvectors_and_rank = self.generate_vector_from_ts(ts_data)
            eigenvectors_list_test = [x[0].iloc[:, :self.min_rank] for x in eigenvectors_and_rank]

            aggregation_df = self.generate_features_from_ts(eigenvectors_list_test, window_mode=self.window_mode)
            aggregation_df = aggregation_df[self.train_feats.columns]

            for col in aggregation_df.columns:
                aggregation_df[col].fillna(value=aggregation_df[col].mean(), inplace=True)

            self.window_length = None
            self.test_feats = None

        return aggregation_df

    def generate_features_from_ts(self, eigenvectors_list: list, window_mode: bool = False) -> pd.DataFrame:

        if window_mode:
            lambda_function_for_stat_features = lambda x: self.apply_window_for_statistical_feature(x.T,
                                                                                                    feature_generator=self.aggregator.create_baseline_features)
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

    def _choose_best_window_size(self, X_train, dataset_name) -> pd.DataFrame:
        """
        Chooses the best window for feature extraction
        :param X_train: train features dataframe
        :param dataset_name: name of the dataset
        :return: dataframe of features extracted with the best window size
        """
        metric_list = []
        n_comp_list = []
        eigen_list = []
        for window_length in self.window_length_list[dataset_name]:
            self.logger.info(f'Generate features for window length - {window_length}')
            self.window_length = window_length

            eigenvectors_and_rank = self.generate_vector_from_ts(X_train.sample(frac=0.5))

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
            self.count = 0
            if self.n_components > 15:
                self.logger.info(f'SSA method find {self.n_components} PCT.'
                                 f'This is mean that SSA method does not find effective low rank structure for '
                                 f'this signal.')
                break

        # index_of_window = int(metric_list.index(max(metric_list)))
        index_of_window = self.pareto_front.pareto_metric_list(np.array(metric_list))
        index_of_window = np.where(index_of_window == True)[0][0]
        self.window_length = self.window_length_list[dataset_name][index_of_window]
        eigenvectors_list = eigen_list[index_of_window]
        self.min_rank = round(np.mean([x.shape[1] for x in eigenvectors_list]))
        self.rank_hyper = self.min_rank
        eigenvectors_and_rank = self.generate_vector_from_ts(X_train)
        self.eigenvectors_list_train = [x[0].iloc[:, :self.min_rank] for x in eigenvectors_and_rank]
        self.train_feats = self.generate_features_from_ts(self.eigenvectors_list_train, window_mode=self.window_mode)
        self.train_feats = self.delete_col_by_var(self.train_feats)
        for col in self.train_feats.columns:
            self.train_feats[col].fillna(value=self.train_feats[col].mean(), inplace=True)

        self.n_components = n_comp_list[index_of_window]
        self.logger.info(f'Window length = {self.window_length} was chosen')

        return self.train_feats

    @staticmethod
    def threading_operation(ts_frame: pd.DataFrame,
                            function_for_feature_extraction: callable):
        pool = Pool(8)
        features = pool.map(function_for_feature_extraction, ts_frame)
        pool.close()
        pool.join()
        return features
