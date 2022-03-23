import os
import pandas as pd
import numpy as np


class ResultsParser:
    """ Class for experiment results parsing """

    def __init__(self):
        self.metrics = ['f1',
                        'roc_auc',
                        'accuracy',
                        'logloss',
                        'precision',
                        ]

        self.root_path = '5min'
        self.comparision_path = '.'

        self.table = pd.DataFrame(columns=['dataset', 'run'] + self.metrics)
        self.fill_table()

    def get_mean_pivot(self):
        return self.table.groupby(['dataset']).mean().round(6)

    def get_pivot(self):
        ...

    def fill_table(self):
        if os.path.exists(self.root_path):
            dataset_folders = [i.split('/')[-1] for i in self.list_dir(self.root_path)]
            index = 0
            for dataset in dataset_folders:
                for run in [i for i in self.list_dir(self.root_path + '/' + dataset) if len(i) < 3]:
                    results = self.read_result(dataset, run).astype(float)
                    self.table.loc[index] = [dataset] + [run] + results.to_list()
                    index += 1
            return
        raise FileNotFoundError('Folder with results experiments is empty or doesnt exists')

    def read_result(self, dataset, version):
        result = pd.read_csv(f'{self.root_path}/{dataset}/{version}/test_results/metrics.csv')['1']

        return result

    def save_to_csv(self, table, name):
        table.to_csv(f'{self.root_path}/{name}.csv')

    def read_mega_table(self, metric: str):
        if metric == 'roc_auc':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='AUROCTest')

        elif metric == 'f1':
            return pd.read_excel(self.comparision_path + '/MegaComparisonResultsSheet.xls', sheet_name='F1Test')

    def get_comparision(self, metric: str):

        mean_results = self.get_mean_pivot()
        mega_table = self.read_mega_table(metric)

        help_dict = {'f1': 'TESTF1',
                     'roc_auc': 'TESTAUROC'}

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

        # ONLY WHERE FEDOT
        mega_table = mega_table[~mega_table['fedot'].isna()]
        # LOOSE and RANK CALCULATION
        for row in mega_table.iterrows():
            dataset_name = row[1][0]
            max_metric = max([value for value in row[1] if type(value) != str])

            fedot_metric = mega_table.loc[mega_table[help_dict[metric]] == dataset_name]['fedot']
            loose = round(100 - fedot_metric * 100 / max_metric, 2)
            mega_table.loc[mega_table[help_dict[metric]] == dataset_name, ['loose_percent']] = loose

        #             metrics = [i for i in row[1] if type(i) != str]
        #             low = []
        #             for i in metrics:
        #                 if i < fedot_metric:
        #                     low.append(i)
        #             rank = str(len(low)) +'/'+str(len(metrics))

        return mega_table

    @staticmethod
    def list_dir(path):
        path_list = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                path_list.append(f)
        return path_list
