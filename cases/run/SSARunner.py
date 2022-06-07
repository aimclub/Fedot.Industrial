from collections import Counter
from core.models.spectral.SSA import Spectrum
from core.models.statistical.Stat_features import AggregationFeatures
from cases.run.ExperimentRunner import ExperimentRunner
from core.operation.utils.utils import *
import matplotlib.pyplot as plt
import timeit


class SSARunner(ExperimentRunner):
    def __init__(self,
                 feature_generanor_dict: dict = None,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None,
                 window_mode: bool = False
                 ):

        super().__init__(feature_generanor_dict, list_of_dataset, launches, metrics_name, fedot_params)
        self.aggregator = AggregationFeatures()
        self.spectrum_extractor = Spectrum

        self.window_length_list = feature_generanor_dict
        if type(self.window_length_list) == int:
            self.window_length_list = [self.window_length_list]

        self.vis_flag = False
        self.rank_hyper = 2
        self.train_feats = None
        self.test_feats = None
        self.n_components = None
        self.window_mode = window_mode

    def __vis_and_save_components(self, Components_df):

        n_rows = round(Components_df[0].shape[1] / 5)

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

        for idx, df in enumerate(Components_df):
            df.plot(subplots=True,
                    figsize=figsize,
                    legend=None,
                    layout=(n_rows + 1, 5))

            plt.tight_layout()
            plt.savefig(os.path.join(self.path_to_save_png, f'components_for_ts_class_{idx}.png'))

    def _ts_chunk_function(self, ts):

        self.logger.info(f'8 CPU on working. '
                         f'Total ts samples - {self.ts_samples_count}. '
                         f'Current sample - {self.count}')

        ts = self.check_Nan(ts)

        spectr = self.spectrum_extractor(time_series=ts,
                                         window_length=self.window_length)
        TS_comps, X_elem, V, Components_df, _, n_components, explained_dispersion = spectr.decompose(
            rank_hyper=self.rank_hyper)

        self.count += 1
        return [Components_df, n_components, explained_dispersion]

    def generate_vector_from_ts(self, ts_frame):
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]
        components_and_vectors = threading_operation(ts_frame=ts_frame.values,
                                                     function_for_feature_exctraction=self._ts_chunk_function)
        self.logger.info(f'Time spent on eigenvectors extraction - {timeit.default_timer() - start}')
        return components_and_vectors

    def extract_features(self, ts_data):

        if self.window_length is None:
            aggregation_df = self._choose_best_window_size(ts_data)
            aggregation_df = delete_col_by_var(self.train_feats)
        else:
            eigenvectors_and_rank = self.generate_vector_from_ts(ts_data)
            eigenvectors_list = [x[0].iloc[:, :self.min_rank] for x in eigenvectors_and_rank]
            aggregation_df = self.generate_features_from_ts(eigenvectors_list, window_mode=self.window_mode)
            aggregation_df = aggregation_df[self.train_feats.columns]

            for col in aggregation_df.columns:
                aggregation_df[col].fillna(value=aggregation_df[col].mean(), inplace=True)

        return aggregation_df

    def generate_features_from_ts(self, eigenvectors_list, window_mode: bool = False):
        start = timeit.default_timer()

        if window_mode:
            lambda_function_for_stat_features = lambda x: apply_window_for_statistical_feature(x.T,
                                                                                               feature_generator=self.aggregator.create_baseline_features)
            lambda_function_for_concat = lambda x: pd.concat(x, axis=1)

            list_with_stat_features_on_interval = list(map(lambda_function_for_stat_features, eigenvectors_list))
            aggregation_df = list(map(lambda_function_for_concat, list_with_stat_features_on_interval))
        else:
            aggregation_df = list(map(lambda x: self.aggregator.create_baseline_features(x.T), eigenvectors_list))

        components_names = aggregation_df[0].index.values
        columns_names = aggregation_df[0].columns.values

        aggregation_df = pd.concat([pd.DataFrame(x.values.ravel()) for x in aggregation_df], axis=1)
        aggregation_df = aggregation_df.T

        new_column_names = []
        for number_of_component in components_names:
            new_column_names.extend([f'{x}_for_component: {number_of_component}' for x in columns_names])

        aggregation_df.columns = new_column_names

        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return aggregation_df

    def _choose_best_window_size(self, X_train):

        metric_list = []
        n_comp_list = []
        disp_list = []
        eigen_list = []

        for window_length in self.window_length_list[self.list_of_dataset[0]]:
            self.logger.info(f'Generate features for window length - {window_length}')
            self.window_length = window_length

            eigenvectors_and_rank = self.generate_vector_from_ts(X_train)

            rank_list = [x[1] for x in eigenvectors_and_rank]
            explained_dispersion = [x[2] for x in eigenvectors_and_rank]

            self.explained_dispersion = round(np.mean(explained_dispersion))

            self.n_components = Counter(rank_list).most_common(n=1)[0][0]

            eigenvectors_list = [x[0].iloc[:, :self.n_components] for x in eigenvectors_and_rank]

            self.logger.info(f'Every eigenvector with impact less then 1 % percent was eliminated. '
                             f'{self.explained_dispersion} % of explained dispersion '
                             f'obtained by first - {self.n_components} components.')

            metrics = self.explained_dispersion  # / self.n_components
            metric_list.append(metrics)

            eigen_list.append(eigenvectors_list)
            n_comp_list.append(self.n_components)
            self.count = 0

        index_of_window = int(metric_list.index(max(metric_list)))
        self.window_length = self.window_length_list[self.list_of_dataset[0]][index_of_window]
        eigenvectors_list = eigen_list[index_of_window]
        self.min_rank = np.min([x.shape[1] for x in eigenvectors_list])
        eigenvectors_list = [x.iloc[:, :self.min_rank] for x in eigenvectors_list]
        self.train_feats = self.generate_features_from_ts(eigenvectors_list, window_mode=self.window_mode)

        for col in self.train_feats.columns:
            self.train_feats[col].fillna(value=self.train_feats[col].mean(), inplace=True)

        self.n_components = n_comp_list[index_of_window]
        self.logger.info(f'Was choosen window length -  {self.window_length}')
        del self.list_of_dataset[0]

        return self.train_feats
