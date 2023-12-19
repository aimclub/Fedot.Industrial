import logging
import os
from abc import ABC
from copy import deepcopy

import pandas as pd

from benchmark.abstract_bench import AbstractBenchmark
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.metrics.metrics_implementation import Accuracy
from fedot_ind.core.repository.constanst_repository import MULTI_CLF_BENCH, UNI_CLF_BENCH


class BenchmarkTSC(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSC, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

        # self._create_output_dir()
        self.experiment_setup = experiment_setup
        self.multi_TSC = MULTI_CLF_BENCH
        self.uni_TSC = UNI_CLF_BENCH
        if custom_datasets is None:
            if use_small_datasets:
                self.custom_datasets = self.uni_TSC
            else:
                self.custom_datasets = self.multi_TSC
        else:
            self.custom_datasets = custom_datasets

        if use_small_datasets:
            self.path_to_result = '/benchmark/results/time_series_uni_clf_comparasion.csv'
        else:
            self.path_to_result = '/benchmark/results/time_series_multi_clf_comparasion.csv'
        self.results_picker = ResultsPicker(path=os.path.abspath(self.output_dir))

    def run(self):
        self.logger.info('Benchmark test started')
        basic_results = self.load_local_basic_results()
        metric_dict = {}
        for dataset_name in self.custom_datasets:
            experiment_setup = deepcopy(self.experiment_setup)
            prediction, target = self.evaluate_loop(dataset_name, experiment_setup)
            metric = Accuracy(target, prediction).metric()
            metric_dict.update({dataset_name: metric})
            basic_results.loc[dataset_name, 'Fedot_Industrial'] = metric
            dataset_path = os.path.join(self.experiment_setup['output_folder'], f'{dataset_name}', 'metrics_report.csv')
            basic_results.to_csv(dataset_path)
        basic_path = os.path.join(self.experiment_setup['output_folder'], 'comprasion_metrics_report.csv')
        basic_results.to_csv(basic_path)
        self.logger.info("Benchmark test finished")

    def load_local_basic_results(self, path: str = None):
        if path is None:
            path = PROJECT_PATH + self.path_to_result
            results = pd.read_csv(path, sep=';', index_col=0)
            results = results.dropna(axis=1, how='all')
            results = results.dropna(axis=0, how='all')
            self.experiment_setup['output_folder'] = PROJECT_PATH + '/benchmark/results/ts_classification'
            return results
        else:
            return self.results_picker.run(get_metrics_df=True, add_info=True)
