import os
import random
from abc import ABC

import pandas as pd

from benchmark.abstract_bench import AbstractBenchmark
from core.api.API import Industrial
from fedot.core.log import default_log as logger
from core.architecture.postprocessing.results_picker import ResultsPicker
from core.ensemble.static.RankEnsembler import RankEnsemble


class BenchmarkTSC(AbstractBenchmark, ABC):
    def __init__(self, number_of_datasets: int = 5, random_selection: bool = False):
        super(BenchmarkTSC, self).__init__(name=self.__class__.__name__,
                                           description="Benchmark for time series classification",
                                           output_dir='./tsc/benchmark_results',
                                           random_selection=random_selection,
                                           number_of_datasets=number_of_datasets)
        self.logger = logger(self.__class__.__name__)
        self._version = '0.0.1'
        self._create_output_dir()
        self.results_picker = ResultsPicker(path=os.path.abspath(self.output_dir))
        self.number_of_datasets = number_of_datasets
        self.random_selection = random_selection

    @property
    def version(self):
        return self._version

    def _get_dataset_list(self, n_samples: int = 5):
        all_datasets = self.results_picker.get_datasets_info()
        dataset_list = self.stratified_ds_selection(all_datasets, n_samples)
        types = []
        for ds in dataset_list:
            types.append(all_datasets[all_datasets['dataset'] == ds]['type'].values[0])

        return dataset_list, types

    @property
    def _config(self):
        dataset_list, types = self._get_dataset_list(n_samples=self.number_of_datasets)
        self.logger.info(f'Selected dataset types: {types}')

        config = {'feature_generator': [
            'window_quantile',
            # 'recurrence',
            'quantile',
            # 'window_spectral',
            # 'spectral',
            # 'wavelet',
            'topological'
        ],

            'datasets_list': dataset_list,
            'use_cache': True,
            'error_correction': False,
            'launches': 1,
            # 'launches': 3,
            'timeout': 1,
            # 'timeout': 5,
            'n_jobs': 2,
            # 'ensemble_algorithm': 'AGG_voting'
        }
        return config

    def run(self):
        self.logger.info('Benchmark test started')
        experimenter = Industrial()
        experimenter.run_experiment(config=self._config,
                                    output_folder=self.output_dir)
        proba_dict, metric_dict = self.results_picker.get_metrics_and_proba()
        rank_ensemble_results = self.apply_rank_ensemble(proba_dict=proba_dict,
                                                         metric_dict=metric_dict)

        basic_report = self.load_basic_results()
        rank_ensemble_report = self.create_report(experiment_results=rank_ensemble_results)
        self.logger.info("Benchmark test finished")

        return basic_report, rank_ensemble_report

    def load_ensemble_results(self, launch_type, model_list: list):
        pass

    def load_basic_results(self):
        return self.results_picker.run(get_metrics_df=True, add_info=True)

    def create_report(self, experiment_results: dict, save_locally: bool = False):
        experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
        experiment_df = experiment_df.fillna(0)
        if 'Best_ensemble_metric' not in experiment_df.columns:
            experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
        experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
        experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
        return experiment_df

    def apply_rank_ensemble(self, proba_dict: dict, metric_dict: dict):
        exp_results = dict()
        for dataset in proba_dict:
            print(f'ENSEMBLE FOR DATASET - {dataset}'.center(50, '–'))
            modelling_results = proba_dict[dataset]
            modelling_metrics = metric_dict[dataset]
            rank_ensemble = RankEnsemble(prediction_proba_dict=modelling_results,
                                         metric_dict=modelling_metrics)

            exp_results.update({dataset: rank_ensemble.ensemble()})
        return exp_results

    def analysis(self):
        # results_table.groupby(['experiment'])['type'].agg(pd.Series.mode) select the most frequent type for generators
        pass

    def stratified_ds_selection(self, df, n_samples=5):
        univariate_tss = df[df['multivariate_flag'] == 0]
        filtered_types = univariate_tss.groupby('type')['type'].count() >= n_samples
        filtered_types = filtered_types[filtered_types].index.tolist()

        univariate_tss = univariate_tss[univariate_tss['type'].isin(filtered_types)]

        if self.random_selection:
            rst = random.randint(0, len(univariate_tss) - 1)
        else:
            rst = 42

        univariate_tss = univariate_tss.groupby('type', group_keys=False).apply(lambda x: x.sample(n_samples,
                                                                                                   random_state=rst))

        return univariate_tss['dataset'].tolist()


if __name__ == "__main__":
    benchmark = BenchmarkTSC(number_of_datasets=1,
                             random_selection=False)
    basic_report, rank_ensemble_report = benchmark.run()

    # benchmark.run()
    res = benchmark.load_ensemble_results(model_list=['quantile', 'recurrence', 'wavelet'], launch_type='max')
