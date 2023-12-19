import logging
import os
from abc import ABC
from copy import deepcopy

import pandas as pd

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.metrics.metrics_implementation import RMSE
from benchmark.abstract_bench import AbstractBenchmark
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker


class BenchmarkTSER(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSER, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

        # self._create_output_dir()
        self.experiment_setup = experiment_setup
        self.monash_regression = [
            "AppliancesEnergy",
            "AustraliaRainfall",
            "BeijingPM10Quality",
            "BeijingPM25Quality",
            "BenzeneConcentration",
            "BIDMC32HR",
            "BIDMC32RR",
            "BIDMC32SpO2",
            "Covid3Month",
            "FloodModeling1",
            "FloodModeling2",
            "FloodModeling3",
            "HouseholdPowerConsumption1",
            "HouseholdPowerConsumption2",
            "IEEEPPG",
            "LiveFuelMoistureContent",
            "NewsHeadlineSentiment",
            "NewsTitleSentiment",
            "PPGDalia",
        ]
        if custom_datasets is None:
            self.custom_datasets = self.monash_regression
        else:
            self.custom_datasets = custom_datasets
        self.use_small_datasets = use_small_datasets
        self.results_picker = ResultsPicker(path=os.path.abspath(self.output_dir))

    def run(self):
        self.logger.info('Benchmark test started')
        basic_results = self.load_local_basic_results()
        metric_dict = {}
        for dataset_name in self.custom_datasets:
            experiment_setup = deepcopy(self.experiment_setup)
            prediction, target = self.evaluate_loop(dataset_name, experiment_setup)
            metric = RMSE(target, prediction).metric()
            metric_dict.update({dataset_name: metric})
            basic_results.loc[dataset_name, 'Fedot_Industrial'] = metric
            dataset_path = os.path.join(self.experiment_setup['output_folder'], f'{dataset_name}', 'metrics_report.csv')
            basic_results.to_csv(dataset_path)
        basic_path = os.path.join(self.experiment_setup['output_folder'], 'comprasion_metrics_report.csv')
        basic_results.to_csv(basic_path)
        self.logger.info("Benchmark test finished")

    def load_local_basic_results(self, path: str = None):
        if path is None:
            path = PROJECT_PATH + '/benchmark/results/time_series_multi_reg_comparasion.csv'
            results = pd.read_csv(path, sep=';', index_col=0)
            results = results.dropna(axis=1, how='all')
            results = results.dropna(axis=0, how='all')
            self.experiment_setup['output_folder'] = PROJECT_PATH + '/benchmark/results/ts_regression'
            return results
        else:
            return self.results_picker.run(get_metrics_df=True, add_info=True)

    # def get_catplot(dataset: pd.DataFrame, metric: str, save_fig: bool = False):
    #     sns.set(font_scale=1.5)
    #     g = sns.catplot(
    #         data=dataset,
    #         kind="bar",
    #         x="experiment", y=metric, hue="type",
    #         palette="tab10", alpha=.6, height=5, aspect=2)
    #     g.despine(left=True)
    #     g.set_axis_labels("", metric)
    #     g.legend.set_title("")
    #     g.fig.suptitle(f'Average {metric.upper()} by experiment type',
    #                    fontsize=14,
    #                    )
    #     if save_fig:
    #         g.savefig(f'./tsc/benchmark_results/{metric}.png')
    #
    #     plt.show()
