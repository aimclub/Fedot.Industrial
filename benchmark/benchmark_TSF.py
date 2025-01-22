import gc
import logging
import os
from abc import ABC
from copy import deepcopy
from typing import Union

import matplotlib
import pandas as pd
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.data.data import InputData
from matplotlib import pyplot as plt

from benchmark.abstract_bench import AbstractBenchmark
from benchmark.feature_utils import DatasetFormatting
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.metrics.metrics_implementation import RMSE, SMAPE
from fedot_ind.core.repository.constanst_repository import (
    M4_FORECASTING_LENGTH,
    MULTI_CLF_BENCH,
    UNI_CLF_BENCH
)
from fedot_ind.tools.loader import DataLoader


class BenchmarkTSF(AbstractBenchmark, ABC):
    def __init__(self,
                 experiment_setup: dict = None,
                 custom_datasets: Union[list, str] = None,
                 use_small_datasets: bool = False):

        super(BenchmarkTSF, self).__init__(
            output_dir='./tser/benchmark_results')
        self.logger = logging.getLogger(self.__class__.__name__)

        self.experiment_setup = experiment_setup
        self.multi_TSC = MULTI_CLF_BENCH
        self.uni_TSC = UNI_CLF_BENCH
        self.automl_TSC = False

        if custom_datasets is None:
            if use_small_datasets:
                self.custom_datasets = self.uni_TSC
            else:
                self.custom_datasets = self.multi_TSC
        else:
            self.custom_datasets = custom_datasets

        if isinstance(self.custom_datasets, str):
            if 'automl' in self.custom_datasets:
                self.automl_TSC = True

        if use_small_datasets:
            self.path_to_result = '/benchmark/results/time_series_uni_forecasts_comparasion.csv'
            self.path_to_save = '/benchmark/results/ts_uni_forecasting'
        else:
            self.path_to_result = '/benchmark/results/m4_results.csv'
            self.path_to_save = '/benchmark/results/ts_uni_forecasting'

        self.results_picker = ResultsPicker(path=os.path.abspath(self.output_dir))
        self.automl_loader = DatasetFormatting()

        # Метаданные для automl
        self.automl_TSC_metadata = None

    def evaluate_loop(self, dataset, experiment_setup: dict = None):
        """
        Если self.automl_TSC=True, dataset - это (file_name, DataFrame).
        Иначе dataset - это строка с именем датасета.
        """
        matplotlib.use('TkAgg')

        if self.automl_TSC:
            file_name, train_data = dataset  # dataset: (str, pd.DataFrame)
            print('-----------', file_name, '-----------')
            row_meta = self.automl_TSC_metadata[self.automl_TSC_metadata['file'] == file_name]
            if row_meta.empty:
                raise ValueError(f"Не найден файл {file_name} в automl_TSC_metadata['file'].")
            forecast_length = int(row_meta['horizon'].iloc[0])
        else:
            file_name = dataset
            train_data = DataLoader(dataset_name=file_name).load_forecast_data()
            forecast_length = M4_FORECASTING_LENGTH.get(file_name, 0)

        if forecast_length <= 0:
            raise ValueError(f"Invalid forecast_length={forecast_length} для датасета {file_name}.")

        if len(train_data) <= forecast_length:
            raise ValueError(f"Insufficient data for {file_name}. "
                             f"Data length ({len(train_data)}) <= forecast_length ({forecast_length}).")

        experiment_setup['task_params'] = TsForecastingParams(forecast_length=forecast_length)

        target = train_data.iloc[-forecast_length:, :].values.ravel()
        train_data = train_data.iloc[:-forecast_length, :].values.ravel()
        print(f"Train data length after split ({file_name}): {len(train_data)}")
        print('train_data shape', 'features:', train_data.shape, 'target:', target.shape)
        
        model = FedotIndustrial(**experiment_setup)
        model.fit((train_data, target))
        prediction = model.predict(train_data)
        plt.close('all')
        return prediction, target, model, file_name

    def run(self, path: str = None):
        self.logger.info('Benchmark test started')
        metric_dict = {}

        if self.automl_TSC and path is not None:
            data_dict = self.load_automl_benchmark(path)
            dataset_list = data_dict.items()  # -> [('economics_1.csv', df), ...]
            basic_results = pd.DataFrame()
        else:
            dataset_list = self.custom_datasets
            basic_results = self.load_local_basic_results()

        for dataset_item in dataset_list:
            experiment_setup = deepcopy(self.experiment_setup)

            # Для automl_TSC=True dataset_item - это (file_name, df)
            if self.automl_TSC:
                prediction, target, model, file_name = self.evaluate_loop(dataset_item, experiment_setup)
                ds_name_for_report = file_name
            else:
                prediction, target, model, file_name = self.evaluate_loop(dataset_item, experiment_setup)
                ds_name_for_report = file_name

            metric = SMAPE(prediction, target).metric()
            metric_dict[ds_name_for_report] = metric
            dataset_path = os.path.join(self.experiment_setup['output_folder'])
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            model_name = experiment_setup['available_operations']
            basic_results.loc[f'{model_name}', ds_name_for_report] = metric
            basic_results.to_csv(os.path.join(dataset_path, 'metrics_report.csv'))

            pred_df = pd.DataFrame({'label': target, 'prediction': prediction})
            pred_df.to_csv(os.path.join(dataset_path, f'prediction_{ds_name_for_report}'), index=False)

            # model.solver.save(dataset_path)  # Если нужно сохранять модель
            gc.collect()

        basic_path = os.path.join(self.experiment_setup['output_folder'], 'comprasion_metrics_report.csv')
        basic_results.to_csv(basic_path)
        self.logger.info("Benchmark test finished")

    def finetune(self):
        self.logger.info('Benchmark finetune started')
        for dataset_name in self.custom_datasets:
            composed_model_path = PROJECT_PATH + self.path_to_save + f'/{dataset_name}' + '/0_pipeline_saved'
            if os.path.isdir(composed_model_path):
                self.experiment_setup['output_folder'] = PROJECT_PATH + self.path_to_save
                experiment_setup = deepcopy(self.experiment_setup)

                prediction, model = self.finetune_loop(dataset_name, experiment_setup)
                metric = RMSE(target, prediction).metric()

                dataset_path = os.path.join(self.experiment_setup['output_folder'],
                                            f'{dataset_name}',
                                            'metrics_report.csv')
                fedot_results = pd.read_csv(dataset_path, index_col=0)

                if dataset_name not in fedot_results.columns:
                    fedot_results[dataset_name] = None
                model_name = model.available_operations
                fedot_results.loc[f'{model_name}_finetuned', dataset_name] = metric
                fedot_results.to_csv(dataset_path)
            else:
                print(f"No composed model for dataset - {dataset_name}")
            gc.collect()
        self.logger.info("Benchmark finetune finished")

    def load_local_basic_results(self, path: str = None):
        path = PROJECT_PATH + self.path_to_result
        results = pd.read_csv(path, sep=',', index_col=0).T
        results = results.dropna(axis=1, how='all')
        results = results.dropna(axis=0, how='all')
        self.experiment_setup['output_folder'] = PROJECT_PATH + self.path_to_save
        return results

    def create_report(self):
        _ = []
        names = []
        for dataset_name in self.custom_datasets:
            model_result_path = PROJECT_PATH + self.path_to_save + f'/{dataset_name}' + '/metrics_report.csv'
            if os.path.isfile(model_result_path):
                df = pd.read_csv(model_result_path, index_col=0, sep=',')
                df = df.fillna(0)
                if 'Fedot_Industrial_finetuned' not in df.columns:
                    df['Fedot_Industrial_finetuned'] = 0
                metrics = df.loc['Fedot_Industrial':'Fedot_Industrial_finetuned', dataset_name]
                _.append(metrics.T.values)
                names.append(dataset_name)
        if len(_) == 0:
            return pd.DataFrame()

        stacked_results = np.stack(_, axis=1).T
        df_res = pd.DataFrame(stacked_results, index=names)
        df_res.columns = ['Fedot_Industrial', 'Fedot_Industrial_finetuned']
        return df_res

    def load_automl_benchmark(self, path):
        load_method_dict = dict(
            automl_univariate=self.automl_loader.format_univariate_forecasting_data,
            automl_global=self.automl_loader.format_global_forecasting_data
        )
        self.automl_TSC_metadata, data_dict = load_method_dict[self.custom_datasets](path, True)
        return data_dict