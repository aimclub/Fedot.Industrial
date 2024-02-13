import gc
import logging
import os
from abc import ABC
from copy import deepcopy

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

from benchmark.abstract_bench import AbstractBenchmark
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.metrics.metrics_implementation import Accuracy, RMSE
from fedot_ind.core.repository.constanst_repository import MULTI_CLF_BENCH, UNI_CLF_BENCH
from fedot_ind.tools.loader import DataLoader


class BenchmarkTSF(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSF, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

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
            self.path_to_result = '/benchmark/results/time_series_uni_forecats_comparasion.csv'
            self.path_to_save = '/benchmark/results/ts_uni_forecasting'
        else:
            self.path_to_result = '/benchmark/results/time_series_multi_forecast_comparasion.csv'
            self.path_to_save = '/benchmark/results/ts_multi_forecasting'
        self.results_picker = ResultsPicker(
            path=os.path.abspath(self.output_dir))

    def evaluate_loop(self, dataset, experiment_setup: dict = None):
        matplotlib.use('TkAgg')
        train_data, test_data = DataLoader(dataset_name=dataset).load_forecast_data()
        target = train_data.iloc[-experiment_setup['task_params'].forecast_length:, :].values.ravel()
        model = FedotIndustrial(**experiment_setup)
        model.fit(train_data)
        prediction = model.predict(train_data)
        plt.close('all')
        return prediction, target

    def run(self):
        self.logger.info('Benchmark test started')
        # basic_results = self.load_local_basic_results()
        metric_dict = {}
        for dataset_name in self.custom_datasets:
            experiment_setup = deepcopy(self.experiment_setup)
            prediction, target = self.evaluate_loop(dataset_name, experiment_setup)
            metric = RMSE(target, prediction).metric()
            pd.DataFrame([prediction, target]).T.plot()
            metric_dict.update({dataset_name: metric})
            dataset_path = os.path.join(self.experiment_setup['output_folder'], f'{dataset_name}',
                                        'metrics_report.csv')
            gc.collect()
        basic_path = os.path.join(self.experiment_setup['output_folder'], 'comprasion_metrics_report.csv')
        self.logger.info("Benchmark test finished")

    def finetune(self):
        self.logger.info('Benchmark finetune started')
        for dataset_name in self.custom_datasets:
            composed_model_path = PROJECT_PATH + self.path_to_save + \
                                  f'/{dataset_name}' + '/0_pipeline_saved'
            if os.path.isdir(composed_model_path):
                self.experiment_setup['output_folder'] = PROJECT_PATH + \
                                                         self.path_to_save
                experiment_setup = deepcopy(self.experiment_setup)
                prediction, target = self.finetune_loop(
                    dataset_name, experiment_setup)
                metric = RMSE(target, prediction).metric()
                dataset_path = os.path.join(self.experiment_setup['output_folder'], f'{dataset_name}',
                                            'metrics_report.csv')
                fedot_results = pd.read_csv(dataset_path, index_col=0)
                fedot_results.loc[dataset_name,
                                  'Fedot_Industrial_finetuned'] = metric

                fedot_results.to_csv(dataset_path)
            else:
                print(f"No composed model for dataset - {dataset_name}")
            gc.collect()
        self.logger.info("Benchmark finetune finished")

    def load_local_basic_results(self, path: str = None):
        path = PROJECT_PATH + self.path_to_result
        results = pd.read_csv(path, sep=',', index_col=0)
        results = results.dropna(axis=1, how='all')
        results = results.dropna(axis=0, how='all')
        self.experiment_setup['output_folder'] = PROJECT_PATH + self.path_to_save
        return results

    def create_report(self):
        _ = []
        names = []
        for dataset_name in self.custom_datasets:
            model_result_path = PROJECT_PATH + self.path_to_save + \
                                f'/{dataset_name}' + '/metrics_report.csv'
            if os.path.isfile(model_result_path):
                df = pd.read_csv(model_result_path, index_col=0, sep=',')
                df = df.fillna(0)
                if 'Fedot_Industrial_finetuned' not in df.columns:
                    df['Fedot_Industrial_finetuned'] = 0
                metrics = df.loc[dataset_name,
                          'Fedot_Industrial':'Fedot_Industrial_finetuned']
                _.append(metrics.T.values)
                names.append(dataset_name)
        stacked_resutls = np.stack(_, axis=1).T
        df_res = pd.DataFrame(stacked_resutls, index=names)
        df_res.columns = ['Fedot_Industrial', 'Fedot_Industrial_finetuned']
        del df['Fedot_Industrial'], df['Fedot_Industrial_finetuned']
        df = df.join(df_res)
        df = df.fillna(0)
        return df
