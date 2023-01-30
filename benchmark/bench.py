import os
import random
from abc import ABC

import pandas as pd
from fedot.core.log import default_log as logger

from benchmark.abstract_bench import AbstractBenchmark
from core.api.API import Industrial
from core.architecture.postprocessing.results_picker import ResultsPicker
from core.architecture.preprocessing.DatasetLoader import DataLoader
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
            'spectral',
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
            'ensemble_algorithm': 'Rank_Ensemble'
        }
        return config

    @property
    def temporary_config(self):
        config = {
            'feature_generator': ['window_quantile',
                                  'quantile',
                                  'topological',
                                  'spectral'],

            'datasets_list': ['Lightning7', 'UMD'],
            'use_cache': True,
            'error_correction': False,
            'launches': 1,
            'timeout': 1,
            'n_jobs': 2,
            'ensemble_algorithm': 'Rank_Ensemble'
        }
        return config

    def run(self):
        self.logger.info('Benchmark test started')
        # experimenter = Industrial()

        # experiment_results = experimenter.run_experiment(config=self._config,
        #                                                  output_folder=self.output_dir)
        #
        # experiment_results = experimenter.run_experiment(config=self.temporary_config,
        #                                                  output_folder=self.output_dir)

        import pickle
        #
        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('filename.pickle', 'rb') as handle:
            experiment_results = pickle.load(handle)


        # proba_dict, metric_dict = self.results_picker.get_metrics_and_proba()
        # ensemble_result_list = self.apply_rank_ensemble(experimenter, experiment_results)

        basic_report = self.load_basic_results()


        # rank_ensemble_report = self.create_report(experiment_results=rank_ensemble_results)
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

    def apply_rank_ensemble(self, experimenter: Industrial, basic_results: dict):
        exp_results = list()
        for dataset in basic_results:
            print(f'ENSEMBLE FOR DATASET - {dataset}'.center(50, '–'))
            ds_results = basic_results[dataset]['Original']
            ds_ensemble_result = experimenter.apply_ensemble(dataset_name=dataset, modelling_results=ds_results)
            if not ds_ensemble_result:
                continue
            ds_ensemble_result.update({'dataset': dataset})
            exp_results.append(pd.DataFrame.from_dict(ds_ensemble_result, orient='index').T)
        try:
            return pd.concat(exp_results, ignore_index=True)
        except ValueError:
            return None

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
                             random_selection=True)
    basic_report, rank_ensemble_report = benchmark.run()
