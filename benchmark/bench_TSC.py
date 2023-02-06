import os
import random
from abc import ABC
from typing import Union

import pandas as pd
import seaborn as sns
from core.log import default_log as logger
# from fedot.core.log import default_log as logger
from matplotlib import pyplot as plt

from benchmark.abstract_bench import AbstractBenchmark
from core.api.API import Industrial
from core.architecture.postprocessing.results_picker import ResultsPicker
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.api.utils.metafeatures import MetaFeaturesDetector


class BenchmarkTSC(AbstractBenchmark, ABC):
    def __init__(self, number_of_datasets: int = 5,
                 random_selection: bool = False,
                 custom_datasets: Union[list, bool] = False):
        super(BenchmarkTSC, self).__init__(
            output_dir='./tsc/benchmark_results',
            random_selection=random_selection,
            number_of_datasets=number_of_datasets)

        self.logger = logger(self.__class__.__name__)
        self._create_output_dir()
        self.number_of_datasets = number_of_datasets
        self.random_selection = random_selection
        self.custom_datasets = custom_datasets

        self.results_picker = ResultsPicker(path=os.path.abspath(self.output_dir))

    def run(self):
        self.logger.info('Benchmark test started')
        experimenter = Industrial()
        experiment_config = self._config

        # Add custom datasets to experiment config if needed
        if self.custom_datasets:
            experiment_config = self._add_custom_datasets(experiment_config)

        experiment_results = experimenter.run_experiment(config=experiment_config,
                                                         output_folder=self.output_dir)
        # with open('filename.pickle', 'rb') as handle:
        #     experiment_results = pickle.load(handle)

        basic_metrics_report, ensemble_report = self._create_report(experiment_results)

        try:
            basic_path = os.path.join(self.output_dir, 'basic_metrics_report.csv')
            basic_metrics_report.to_csv(basic_path, index=False)

            ensemble_path = os.path.join(self.output_dir, 'ensemble_report.csv')
            ensemble_report.to_csv(ensemble_path, index=False)
        except Exception as ex:
            self.logger.error(f'Can not save report: {ex}')

        # local_report = self._create_local_report()

        self.logger.info("Benchmark test finished")

    @property
    def _config(self):
        dataset_list, types = self._get_dataset_list(n_samples=self.number_of_datasets)
        self.logger.info(f'Selected types: {types}')

        # dataset_list = ['Car']
        config = {'feature_generator': [
                                        'window_quantile',
                                        'recurrence',
                                        'quantile',
                                        'window_spectral',
                                        'spectral',
                                        'wavelet',
                                        'topological'
                                        ],
                  'datasets_list': dataset_list, 'use_cache': False,
                  'error_correction': False, 'launches': 2,
                  'timeout': 15, 'n_jobs': 2,
                  'ensemble_algorithm': 'Rank_Ensemble'
                  }
        return config

    def _get_dataset_list(self, n_samples):
        all_datasets = self.results_picker.get_datasets_info()
        dataset_list = self.stratified_ds_selection(all_datasets, n_samples)
        types = []
        for ds in dataset_list:
            types.append(all_datasets[all_datasets['dataset'] == ds]['type'].values[0])

        return dataset_list, types

        # return ['ItalyPowerDemand', 'UMD', 'Coffee', 'GunPoint']

    def _create_report(self, experiment_results):
        metrics_df = self._get_basic_results_table(experiment_results)
        ensemble_df = self._get_ensemble_results_table(experiment_results)

        datasets_info = ResultsPicker().get_datasets_info()

        basic_metrics_report = pd.merge(metrics_df, datasets_info, how='left', on='dataset')
        if ensemble_df is not None:
            ensemble_report = pd.merge(ensemble_df, datasets_info, how='left', on='dataset')
        else:
            ensemble_report = None

        if basic_metrics_report.isnull().values.any():
            basic_metrics_report = self._fill_na_metafeatures(basic_metrics_report)
            ensemble_report = self._fill_na_metafeatures(ensemble_report)

        self.basic_analysis(basic_metrics_report, save_locally=True)
        self.ensemble_analysis(ensemble_report, save_locally=True)

        return basic_metrics_report, ensemble_report

    def _create_local_report(self):
        df = self.load_local_basic_results()
        self.basic_analysis(df, save_locally=True)
        return df

    def load_local_basic_results(self):
        return self.results_picker.run(get_metrics_df=True, add_info=True)

    def _fill_na_metafeatures(self, basic_metrics_report: pd.DataFrame):
        """ Fill missing meta-data about datasets

        Args:
            basic_metrics_report: report with basic metrics

        """
        if basic_metrics_report is None:
            return None

        datasets_with_na = basic_metrics_report[basic_metrics_report.isnull().any(axis=1)]['dataset'].unique()
        for dataset in datasets_with_na:
            train_data, test_data = DataLoader(dataset).load_data()
            detector = MetaFeaturesDetector(train_data=train_data, test_data=test_data, dataset_name=dataset)
            base_meta_features = detector.get_base_metafeatures()

            index_of_na = basic_metrics_report[basic_metrics_report['dataset'] == dataset].index.tolist()
            for ind in index_of_na:
                for key, value in base_meta_features.items():
                    basic_metrics_report.loc[ind, key] = value

        return basic_metrics_report

    def _get_basic_results_table(self, experiment_results):
        basic_result_df = pd.DataFrame()
        for dataset in experiment_results.keys():
            try:
                for generator in experiment_results[dataset]['Original']:
                    for launch in experiment_results[dataset]['Original'][generator].keys():
                        metrics = experiment_results[dataset]['Original'][generator][launch]['metrics']
                        basic_result_df = basic_result_df.append(
                            {'dataset': dataset, 'experiment': generator, 'f1': metrics['f1'],
                             'roc_auc': metrics['roc_auc'], 'accuracy': metrics['accuracy'],
                             'precision': metrics['precision'], 'logloss': metrics['logloss']},
                            ignore_index=True)
            except TypeError:
                continue

        # Ensure that `dataset` and `experiment` columns are the first two columns
        cols = basic_result_df.columns.tolist()
        cols.pop(cols.index('dataset'))
        cols.pop(cols.index('experiment'))
        metrics_df = basic_result_df[['dataset', 'experiment'] + cols]
        return metrics_df

    def _get_ensemble_results_table(self, experiment_results):

        # with open('ensemble_report_example.pickle', 'rb') as handle:
        #     ensemble_df = pickle.load(handle)
        #     # ensemble_df = ensemble_df.reset_index()
        #     # cols = list(ensemble_df.columns)
        #     # cols[0] = 'dataset'
        #     # ensemble_df.columns = cols
        # return ensemble_df

        ensemble_path = os.path.join(self.output_dir, 'ensemble')
        if os.path.exists(ensemble_path):
            files = self.results_picker.list_files(ensemble_path)
            ls = [pd.read_csv(os.path.join(ensemble_path, files[0]), index_col=0) for file in files]

        else:
            self.logger.info('No ensemble results found')
            return None

        return pd.concat(ls, ignore_index=True)

    def basic_analysis(self, report: pd.DataFrame, save_locally: bool = False):
        data_binary = report[report['number_of_classes'] == 2].reset_index(drop=True)
        data_multi = report[report['number_of_classes'] > 2].reset_index(drop=True)

        self.get_catplot(data_binary, 'roc_auc', save_fig=save_locally)
        self.get_catplot(data_multi, 'f1', save_fig=save_locally)

    def ensemble_analysis(self, ensemble_report, save_locally):
        """
        Now just a plug before I figure out how to analyze ensemble results

        Args:
            ensemble_report: report with ensemble results
            save_locally: save plots locally

        Returns:

        """
        if save_locally and ensemble_report is not None:
            return ensemble_report

    @staticmethod
    def get_catplot(dataset: pd.DataFrame, metric: str, save_fig: bool = False):
        sns.set(font_scale=1.5)
        g = sns.catplot(
            data=dataset,
            kind="bar",
            x="experiment", y=metric, hue="type",
            palette="tab10", alpha=.6, height=5, aspect=2)
        g.despine(left=True)
        g.set_axis_labels("", metric)
        g.legend.set_title("")
        g.fig.suptitle(f'Average {metric.upper()} by experiment type',
                       fontsize=14,
                       )
        if save_fig:
            g.savefig(f'./tsc/benchmark_results/{metric}.png')

        plt.show()

    def stratified_ds_selection(self, all_datasets_table: pd.DataFrame, n_samples: int = 5):
        """
        Selects n_samples !!! OF SMALL !!! datasets from each type
        Args:
            all_datasets_table: pd.DataFrame with all datasets info (from results_picker)
            n_samples: number of datasets to select from each type

        Returns:
            list of selected datasets names

        """
        univariate_tss = all_datasets_table[all_datasets_table['multivariate_flag'] == 0]

        # TODO: DELETE THIS: SMALL SERIES ONLY!!!!
        univariate_tss = univariate_tss[(univariate_tss['train_size'] < 1000) & (univariate_tss['length'] < 1000)]

        filtered_by_type_quantity = univariate_tss.groupby('type')['type'].count() >= n_samples
        filtered_types = filtered_by_type_quantity[filtered_by_type_quantity].index.tolist()

        univariate_tss = univariate_tss[univariate_tss['type'].isin(filtered_types)]

        if self.random_selection:
            rst = random.randint(0, len(univariate_tss) - 1)
        else:
            rst = 42

        univariate_tss = univariate_tss.groupby('type', group_keys=False).apply(lambda x: x.sample(n_samples,
                                                                                                   random_state=rst))

        return univariate_tss['dataset'].tolist()

    def _add_custom_datasets(self, experiment_config):
        for ds in self.custom_datasets:
            experiment_config['datasets_list'].append(ds)
        self.logger.info(f'Custom datasets added: {self.custom_datasets}')
        return experiment_config


if __name__ == "__main__":

    n_datasets = 1
    rnd_select = True

    bnch = BenchmarkTSC(number_of_datasets=n_datasets,
                        random_selection=rnd_select,
                        custom_datasets=False
                        # custom_datasets=['Lightning7_fake']
                        )
    bnch.run()
