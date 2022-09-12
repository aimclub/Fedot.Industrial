import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.operation.utils.utils import PROJECT_PATH

dataset_types = {
    'equal': ['Trace', 'ShapesAll', 'Beef', 'DodgerLoopDay', 'ScreenType', 'Lightning7', 'EigenWorms',
              'Plane', 'EthanolLevel', 'AsphaltRegularity', 'ElectricDevices', 'PowerCons', 'Herring'],

    'high_train': ['Worms', 'CounterMovementJump', 'FordA', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                   'FingerMovements', 'HandOutLines', 'FordB', 'SpokenArabicDigits', 'DistalPhalanxOutlineAgGroup',
                   'Tiselac', 'Earthquakes'],

    'low_train': ['Chinatown', 'ItalyPowerDemand', 'Crop', 'MoteStrain', 'DodgerLoopWeekend', 'ECG5000', 'ArrowHead',
                  'ChlorineConcentration', 'GunPoint', 'DodgerLoopGame', 'BME', 'DiatomSizeReduction', 'CinCECGTorso',
                  'Haptics', 'Rock']
}


class ResultsParser:
    """ Class for experiment results parsing """

    def __init__(self):
        self.metrics_dict = {'f1': 'TESTF1',
                             'roc_auc': 'TESTAUROC',
                             'accuracy': 'TESTACC',
                             'logloss': 'TESTNLL',
                             'precision': 'TESTPrec'
                             }
        self.timeout = '1_hour'
        self.results_path = os.path.join(PROJECT_PATH,
                                         'results_of_experiments',
                                         self.timeout)
        self.comparison_path = os.path.join(PROJECT_PATH, 'results_of_experiments')
        self.table = pd.DataFrame(columns=['dataset', 'run'] + list(self.metrics_dict.keys()))
        self.fill_table()

    def get_mean_pivot(self):
        """
        Create pivot table with mean values of metrics
        """
        return self.table.groupby(['dataset']).mean().round(6)

    def fill_table(self):
        """
        Function for parsing cases results into single table
        """
        if os.path.exists(self.results_path):
            dataset_folders = [i.split('/')[-1] for i in self.list_dir(self.results_path)]
            index = 0
            for dataset in dataset_folders:
                for run in [i for i in self.list_dir(os.path.join(self.results_path, dataset)) if len(i) < 3]:
                    results = self.read_result(dataset, run)
                    if results is not None:
                        results = results.astype(float)
                    else:
                        continue
                    self.table.loc[index] = [dataset] + [run] + results.to_list()
                    index += 1
            return
        raise FileNotFoundError('Folder with results cases is empty or doesnt exists')

    def read_result(self, dataset: str, version: str):
        """
        Function to parse a single result
        :param dataset: name of dataset
        :param version: name of run
        """
        try:
            result = pd.read_csv(f'{self.results_path}/{dataset}/{version}/test_results/metrics.csv')['1']
            return result
        except FileNotFoundError or FileExistsError:
            return None

    def save_to_csv(self, table_object, name):
        table_object.to_csv(f'{self.results_path}/{name}.csv')

    def read_mega_table(self, metric: str):
        """
        Function for parsing table with results of cases with different algorithms

        :param metric: name of metric to extract from mega table (e.g. f1, roc_auc)
        :return specific (by metric) sheet of excel-table converted to pd.DataFrame
        """
        sheet_lists = {'roc_auc': 'AUROCTest',
                       'f1': 'F1Test',
                       'accuracy': 'ACCTest',
                       'logloss': 'NLLTest',
                       'precision': 'PrecTest'}

        table_name = 'MegaComparisonResultsSheet.xls'
        return pd.read_excel(os.path.join(self.comparison_path, table_name), sheet_name=sheet_lists[metric])

    def get_comparison(self, metric: str, full_table: bool = True):
        """
        Function for comparison FEDOT results with other algorithms by chosen metric

        :param full_table: True (default) returns full table with all datasets. False returns shortened version
        :param metric: name of metric to compare by (e.g. f1, roc_auc)
        :return: slice of mega table where FEDOT result is present
        """
        mean_results = self.get_mean_pivot()
        results_mega_table = self.read_mega_table(metric)
        metric_name_in_megatable = self.metrics_dict[metric]

        mega_table = results_mega_table.copy()
        for new_col in ['fedot', 'outperformed_by_fedot', 'loose_percent']:
            mega_table.insert(len(mega_table.loc[0]), new_col, np.NaN)

        for row in mean_results.iterrows():
            ds_name = row[0]
            metric_value = row[1][metric]

            if ds_name in mega_table[metric_name_in_megatable].to_list():
                mega_table.loc[mega_table[metric_name_in_megatable] == ds_name, ['fedot']] = metric_value
            else:
                index = len(mega_table)
                length = len(mega_table.loc[0])
                mega_table.loc[index] = [ds_name] + [np.NaN for _ in range(length - 4)] + [metric_value] + [np.NaN] * 2

        # get comparison table where fedot is
        mega_table = mega_table[~mega_table['fedot'].isna()]

        best_algos = []
        best_metrics = []

        # LOOSE CALCULATION
        for index, row in mega_table.iterrows():
            dataset_name = row[0]

            max_metric = max([value for value in mega_table.loc[index][1:16] if str(value) != 'nan'])
            best_candidate = row[1:16].astype(float).idxmax()

            if row[1:16][best_candidate] == row[1:16]['fedot']:
                best_algorithm = 'fedot'
            else:
                best_algorithm = best_candidate

            best_algos.append(best_algorithm)
            best_metrics.append(max_metric)

            fedot_metric = float(mega_table.loc[mega_table[metric_name_in_megatable] == dataset_name]['fedot'])
            loose = abs(round(100 - fedot_metric * 100 / max_metric, 2))
            mega_table.loc[mega_table[metric_name_in_megatable] == dataset_name, ['loose_percent']] = loose

            metrics = [value for value in mega_table.loc[index][1:15]]
            lower_fedot = []
            for i in metrics:
                if i <= fedot_metric or i == fedot_metric:
                    lower_fedot.append(i)
            rank = str(len(lower_fedot)) + '/' + str(len(metrics))
            mega_table.loc[mega_table[metric_name_in_megatable] == dataset_name, ['outperformed_by_fedot']] = rank

        dataset_type_list = []
        for row in mega_table.iterrows():
            dataset_name = row[1][0]

            types = dataset_types.keys()
            for ds_type in types:
                if dataset_name in dataset_types[ds_type]:
                    dataset_type_list.append(ds_type)

        mega_table['best_algo'] = best_algos
        mega_table['max_metric'] = best_metrics
        mega_table['dataset_types'] = dataset_type_list

        if full_table:
            return mega_table
        return mega_table[[f'{self.metrics_dict[metric]}', 'fedot', 'outperformed_by_fedot',
                           'loose_percent', 'dataset_types', 'best_algo', 'max_metric']]

    def get_compare_boxplots(self, metrics: list = ('f1', 'roc_auc')):
        for metric in metrics:
            sns.set(font_scale=1.5)
            fig, ax = plt.subplots(figsize=(12, 6))
            y = 'dataset'
            x = metric
            sns.boxplot(data=self.table,
                        y=y,
                        x=x,
                        palette="pastel",
                        width=0.7,
                        showmeans=False
                        )
            sns.swarmplot(data=self.table, y=y, x=x, color=".25")
            plt.xlabel(f'{metric.upper()} Metric', fontsize=20)
            plt.ylabel('', fontsize=20)

            save_path = os.path.join(self.results_path, f'{metric}_boxplot_{self.timeout}.png')
            plt.savefig(fname=save_path, dpi=320, bbox_inches='tight')

    @staticmethod
    def list_dir(path):
        """
        Function used instead of os.listdir() to get list of non-hidden directories
        :param path: path to scan
        :return: list of available directories
        """
        path_list = []
        for f in os.listdir(path):
            if '.' not in f:
                path_list.append(f)
        return path_list


if __name__ == '__main__':
    # Example of usage
    metrics = ['f1', 'roc_auc']
    parser = ResultsParser()

    for metr in metrics:
        table = parser.get_comparison(metr, full_table=False)
        parser.get_compare_boxplots()
        parser.save_to_csv(table, f'{metr}_mega_compare_{parser.timeout}')
