import os
import pandas as pd
import numpy as np

dataset_types = {
    'equal': ['Trace', 'ShapesAll', 'Beef', 'DodgerLoopDay', 'ScreenType', 'Lightning7', 'EigenWorms', 'Rock',
              'Plane', 'EthanolLevel', 'AsphaltRegulatiry', 'ElectricDevices', 'PowerCons', 'Herring'],
    'high_train': ['Worms', 'CounterMovementJump', 'FordA', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                   'FingerMovements', 'HandOutLines', 'FordB', 'SpokenArabicDigits', 'DistalPhalanxOutlineAgGroup',
                   'Tiselac', 'Earthquakes'],
    'low_train': ['Chinatown', 'ItalyPowerDemand', 'Crop', 'MoteStrain', 'DodgerLoopWeekend', 'ECG5000', 'ArrowHead',
                  'ChlorineConcentration', 'GunPoint', 'DodgerLoopGame', 'BME', 'DiatomSizeReduction', 'CinCECGTorso',
                  'Haptics']
}


class ResultsParser:
    """ Class for experiment results parsing """

    def __init__(self):
        self.metrics = ['f1',
                        'roc_auc',
                        'accuracy',
                        'logloss',
                        'precision',
                        ]

        self.root_path = '../results_of_experiments/20_min'
        self.comparision_path = '../results_of_experiments/'

        self.table = pd.DataFrame(columns=['dataset', 'run'] + self.metrics)
        self.fill_table()

    def get_mean_pivot(self):
        """ Create pivot table with mean values of metrics """
        return self.table.groupby(['dataset']).mean().round(6)

    def fill_table(self):
        """ Function for parsing experiments results into single table """
        if os.path.exists(self.root_path):
            dataset_folders = [i.split('/')[-1] for i in self.list_dir(self.root_path)]
            index = 0
            for dataset in dataset_folders:
                for run in [i for i in self.list_dir(self.root_path + '/' + dataset) if len(i) < 3]:
                    results = self.read_result(dataset, run)
                    if results is not None:
                        results = results.astype(float)
                    else:
                        continue

                    self.table.loc[index] = [dataset] + [run] + results.to_list()
                    index += 1
            return
        raise FileNotFoundError('Folder with results experiments is empty or doesnt exists')

    def read_result(self, dataset, version):
        """ Function to parse a single result """
        try:
            result = pd.read_csv(f'{self.root_path}/{dataset}/{version}/test_results/metrics.csv')['1']
            return result
        except Exception:
            return None

    def save_to_csv(self, table, name):
        table.to_csv(f'{self.root_path}/{name}.csv')

    def read_mega_table(self, metric: str):
        """ Function for parsing table with results of experiments with different algorithms

        :param metric: name of metric to extract from mega table (e.g. f1, roc_auc)
        :return specific (by metric) sheet of excel-table converted to pd.DataFrame
        """
        if metric == 'roc_auc':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='AUROCTest')

        elif metric == 'f1':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='F1Test')

        elif metric == 'accuracy':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='ACCTest')

        elif metric == 'logloss':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='NLLTest')

        elif metric == 'precision':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='PrecTest')

    def get_comparison(self, metric: str):
        """ Function for comparison FEDOT results with other algorithms by chosen metric

        :param metric: name of metric to compare by (e.g. f1, roc_auc)
        :return: slice of mega table where FEDOT result is present
        """
        mean_results = self.get_mean_pivot()
        results_mega_table = self.read_mega_table(metric)

        help_dict = {'f1': 'TESTF1',
                     'roc_auc': 'TESTAUROC',
                     'accuracy': 'TESTACC',
                     'logloss': 'TESTNLL',
                     'precision': 'TESTPrec'
                     }

        mega_table = results_mega_table.copy()
        mega_table[['fedot', 'outperformed_by_fedot', 'loose_percent']] = np.NaN

        for row in mean_results.iterrows():
            name = row[0]
            metric_value = row[1][metric]

            if name in mega_table[help_dict[metric]].to_list():

                mega_table.loc[mega_table[help_dict[metric]] == name, ['fedot']] = metric_value

            else:
                index = len(mega_table)
                length = len(mega_table.loc[0])

                mega_table.loc[index] = [name] + [np.NaN for _ in range(length - 4)] + [metric_value] + [np.NaN, np.NaN]

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

            fedot_metric = mega_table.loc[mega_table[help_dict[metric]] == dataset_name]['fedot']
            loose = abs(round(100 - fedot_metric * 100 / max_metric, 2))
            mega_table.loc[mega_table[help_dict[metric]] == dataset_name, ['loose_percent']] = loose

        # RANK CALCULATION
        for index, row in mega_table.iterrows():
            dataset_name = row[0]
            metrics = [value for value in mega_table.loc[index][1:15]]

            fedot_metric = float(mega_table.loc[mega_table[help_dict[metric]] == dataset_name]['fedot'])

            lower_fedot = []
            for i in metrics:
                if i <= fedot_metric or i == fedot_metric:
                    lower_fedot.append(i)

            rank = str(len(lower_fedot)) + '/' + str(len(metrics))
            mega_table.loc[mega_table[help_dict[metric]] == dataset_name, ['outperformed_by_fedot']] = rank

        mega_table['dataset_types'] = np.NaN
        for row in mega_table.iterrows():
            dataset_name = row[1][0]
            for dataset_type, value in dataset_types.items():
                for ds in value:
                    if ds == dataset_name:
                        mega_table.loc[mega_table[help_dict[metric]] == dataset_name, ['dataset_types']] = dataset_type

        mega_table['best_algo'] = best_algos
        mega_table['max_metric'] = best_metrics

        return mega_table

    @staticmethod
    def list_dir(path):
        """
        Function used instead of os.listdir() to get list of non-hidden directories
        :param path: path to scan
        :return: list of available directories
        """
        path_list = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                path_list.append(f)
        return path_list


# EXAMPLE

ls = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']
c = ResultsParser()

for metr in ls:
    table = c.get_comparison(metr)

    c.save_to_csv(table, f'{metr}_mega_compare_20min')
