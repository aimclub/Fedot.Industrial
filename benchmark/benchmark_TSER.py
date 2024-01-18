import logging
import os
from abc import ABC
from copy import deepcopy

import matplotlib

from fedot_ind.tools.loader import DataLoader

matplotlib.use('TkAgg')
import pandas as pd
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.metrics.metrics_implementation import RMSE
from benchmark.abstract_bench import AbstractBenchmark
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.repository.constanst_repository import MULTI_REG_BENCH


class BenchmarkTSER(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: list = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSER, self).__init__(
            output_dir='./tser/benchmark_results')

        self.logger = logging.getLogger(self.__class__.__name__)

        self.experiment_setup = experiment_setup
        self.monash_regression = MULTI_REG_BENCH
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

    def finetune(self):
        for dataset_name in self.custom_datasets:
            experiment_setup = deepcopy(self.experiment_setup)
            composed_model_path = PROJECT_PATH + '/benchmark/results/ts_regression' + f'/{dataset_name}' + '/0_pipeline_saved'
            experiment_setup['output_folder'] = composed_model_path
            del experiment_setup['industrial_preprocessing']
            prediction, model = self.finetune_loop(dataset_name, experiment_setup)
            metric = RMSE(model.predict_data.target, prediction).metric()
            metric_path = PROJECT_PATH + '/benchmark/results/ts_regression' + f'/{dataset_name}' + '/metrics_report.csv'
            fedot_results = pd.read_csv(metric_path, index_col=0)
            fedot_results.loc[dataset_name, 'Fedot_Industrial_finetuned'] = metric
            fedot_results.to_csv(metric_path)
            model.save_best_model()
        self.logger.info("Benchmark finetune finished")

    def finetune_loop(self, dataset, experiment_setup: dict = None):
        train_data, test_data = DataLoader(dataset_name=dataset).load_data()
        model = FedotIndustrial(**experiment_setup)
        model.load(path=experiment_setup['output_folder'])
        model.finetune(train_data, tuning_params=experiment_setup, mode='head')
        prediction = model.finetune_predict(test_data)
        return prediction, model

    def show_composite_pipeline(self):
        for dataset_name in self.custom_datasets:
            composed_model_path = PROJECT_PATH + '/benchmark/results/ts_regression' + f'/{dataset_name}' + '/0_pipeline_saved'
            experiment_setup = deepcopy(self.experiment_setup)
            experiment_setup['output_folder'] = composed_model_path
            del experiment_setup['industrial_preprocessing']
            model = FedotIndustrial(**experiment_setup)
            model.load(path=composed_model_path)
            batch_pipelines = [automl_branch.fitted_operation.model.current_pipeline
                               for automl_branch in model.current_pipeline.nodes if automl_branch.name == 'fedot_regr']
            pr = PipelineNode('ridge', nodes_from=[p.root_node for p in batch_pipelines])
            composed_pipeline = Pipeline(pr)
            composed_pipeline.show()
