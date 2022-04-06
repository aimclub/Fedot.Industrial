import os
from functools import reduce
import pandas as pd
from core.operation.utils.utils import path_to_save_results


class TabularReporter:
    """ Class for preparing reports and visualisations """

    def __init__(self, path_to_save: str = path_to_save_results(), launches: int = 1):
        self.path_to_save = path_to_save
        self.launches = launches

    def get_final_pipeline_paths(self, path: str):
        table_by_generation = dict()
        metrics_by_generation = dict()
        generation = os.listdir(path)
        generation = [x for x in generation if not x.endswith('txt')]
        dataset_files = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        dataset_dirs = list(filter(lambda x: os.path.isdir(x), dataset_files))
        return table_by_generation, metrics_by_generation

    def create_report(self,
                      launches: int = 1,
                      save_flag: bool = False):

        df_list = []
        dataset_generation_history = dict()
        metric_generation_history = dict()

        path = os.path.join(self.path_to_save, library)
        names = os.listdir(path)
        dataset_list = []
        for data_name in names:
            launch_list = []
            for launch in range(launches):
                dataset_abs_path = os.path.join(path, data_name, f'launch_{launch}', 'test_results')
                if not os.path.exists(dataset_abs_path):
                    print(f'No results of experiment for dataset - {data_name}')
                    break
                else:
                    df = pd.read_csv(dataset_abs_path + '/metrics.csv')
                    df = df.iloc[:, 2:]
                    df.columns.values[:2] = ['Metric_name', f'Metric_{library}_value_launch_{launch}']
                    df['Dataset'] = data_name
                    df.columns = ['Metric_name', f'Metric_{library}_value_launch_{launch}',
                                      f'Inference_{library}_launch_{launch}',
                                      f'Fit_time_{library}_launch_{launch}', 'Dataset']
                    df = df.drop_duplicates()
                    launch_list.append(df)

                if len(launch_list) == 0:
                    print(f'No metric for_{data_name}')
                else:
                    dataset_report = pd.concat(launch_list, axis=1)
                    dataset_report = dataset_report.loc[:, ~dataset_report.columns.duplicated()]
                    dataset_report_metric = dataset_report[
                        [x for x in dataset_report.columns if x.startswith(f'Metric_{library}')]]
                    dataset_report_inference = dataset_report[
                        [x for x in dataset_report.columns if x.startswith(f'Inference_{library}')]]
                    dataset_report[f'Mean_Metric_{library}_value'] = dataset_report_metric.mean(axis=1)
                    dataset_report[f'Std_Metric_{library}_value'] = dataset_report_metric.std(axis=1)
                    dataset_report[f'Mean_Inference_{library}'] = dataset_report_inference.mean(axis=1)
                    dataset_report[f'Std_Inference_{library}'] = dataset_report_inference.std(axis=1)
                    dataset_list.append(dataset_report)
            df_report = pd.concat(dataset_list)
            df_list.append(df_report)
        merged_df = reduce(lambda x, y: pd.merge(x, y, on=['Dataset', 'Metric_name'], how='outer'), df_list)
        merged_df = merged_df.fillna(0)
        report = merged_df.groupby(by=['Dataset', 'Metric_name']).first().apply(lambda x: round(x, 3))
        ff = report[[x for x in report.columns if x.startswith('Mean') or x.startswith('Std')]]
        if save_flag:
            for lib in self.libraries_to_compare:
                csv_name = f'{lib}_vs_{lib}'
            report.to_csv(f'./report_comparison_{csv_name}.csv')

        return merged_df
