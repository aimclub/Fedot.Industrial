import os
import pandas as pd


class ResultsParser:
    """ Class for experiment results parsing """

    def __init__(self):
        self.metrics = ['f1',
                        'roc_auc',
                        'accuracy',
                        'logloss',
                        'precision']
        self.root_path = '../results_of_experiments'
        self.table = pd.DataFrame(columns=['dataset', 'run'] + self.metrics)
        self.fill_table()

    def get_mean_pivot(self):
        return self.table.groupby(['dataset']).mean().round(2)

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
        return pd.read_csv(f'{self.root_path}/{dataset}/{version}/test_results/metrics.csv')['1']

    def save_to_csv(self, table, name):
        table.to_csv(f'{self.root_path}/{name}.csv')

    @staticmethod
    def list_dir(path):
        path_list = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                path_list.append(f)
        return path_list
