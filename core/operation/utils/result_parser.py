import os
from typing import Union

import numpy as np
import pandas as pd

from core.operation.utils.utils import PROJECT_PATH


class ResultsParser:
    """Class for parsing results of experiments. It parses all experiments in ``results_of_experiments``
    folder and creates dataframe table that availible for further analysis.

    Examples:
        >>> parser = ResultsParser()
        >>> results = parser.run()

    """

    def __init__(self):
        self.exp_path = os.path.join(PROJECT_PATH, 'results_of_experiments')
        self._metrics = ('f1', 'roc_auc')
        self.exp_folders = ['window_spectral',
                            'topological',
                            'recurrence',
                            'ensemble',
                            'quantile',
                            'spectral',
                            'wavelet',
                            'window_quantile']
        self.ds_info_path = os.path.join(PROJECT_PATH, 'core/operation/utils/ds_info.plk')
        # self.ds_info_path = os.path.join(PROJECT_PATH, 'core/operation/utils/ds_info.csv')

    def run(self) -> pd.DataFrame:
        """
        Base method for parsing results of experiments.

        Returns:
            Table with results of experiments.

        """
        ds_info = pd.read_pickle(self.ds_info_path)
        # ds_info = pd.read_csv(self.ds_info_path, index_col=0)
        exp_results = self.retrieve_data()
        ls = []
        for ds in exp_results['dataset']:
            ls.append(int(ds_info[ds_info['dataset'] == ds]['n_classes']))
        exp_results['n_classes'] = ls

        return exp_results

    def read_proba(self, launch: Union[int, str] = 1, path=None, exp_folders: list = None):
        if exp_folders is None:
            exp_folders = self.exp_folders
        current = []
        proba_dict = {}
        metric_dict = {}
        filtered_folders = []
        for folder in exp_folders:
            try:
                path_to_results = os.path.join(self.exp_path, path, folder)
                datasets = os.listdir(path_to_results)
                current.append(datasets)
                filtered_folders.append(folder)
            except Exception:
                print(f'{folder} is empty')
        current = sorted(current, key=lambda x: len(x), reverse=True)
        dataset_path = [list(filter(lambda x: x in current[0], sublist)) for sublist in current[1:]]
        _ = []
        for pth in dataset_path:
            _.extend(pth)
        dataset_filtered = list(set(_))
        for filtered in filtered_folders:
            for dataset in dataset_filtered:
                try:
                    if dataset not in proba_dict.keys():
                        proba_dict[dataset] = {}
                    if dataset not in metric_dict.keys():
                        metric_dict[dataset] = {}
                    if type(launch) == str:
                        best_metric = 0
                        launch = 1
                        for dir in os.listdir(os.path.join(self.exp_path, path, filtered, dataset)):
                            metric_path = os.path.join(self.exp_path, path, filtered, dataset, dir, 'test_results',
                                                       'metrics.csv')
                            metrics = pd.read_csv(metric_path, index_col=0)
                            if 'index' in metrics.columns:
                                del metrics['index']
                                metrics = metrics.T
                                metrics = metrics.rename(columns=metrics.iloc[0])
                                metrics = metrics[1:]
                            metric_sum = metrics['roc_auc'].values[0] + metrics['f1'].values[0]
                            if metric_sum > best_metric:
                                best_metric = metric_sum
                                launch = dir

                    metric_path = os.path.join(self.exp_path, path, filtered, dataset, str(launch), 'test_results',
                                                   'metrics.csv')
                    proba_path = os.path.join(self.exp_path, path, filtered, dataset, str(launch), 'test_results',
                                                  'probs_preds_target.csv')
                    proba_dict[dataset].update({filtered: pd.read_csv(proba_path, index_col=0)})
                    metrics = pd.read_csv(metric_path, index_col=0)
                    if 'index' in metrics.columns:
                        del metrics['index']
                        metrics = metrics.T
                        metrics = metrics.rename(columns=metrics.iloc[0])
                        metrics = metrics[1:]
                    metric_dict[dataset].update({filtered: metrics})
                except Exception:
                    print(f'{folder} and {dataset} doesnt exist.')
        return proba_dict, metric_dict

    def retrieve_data(self):
        experiments = self.list_dir(self.exp_path)
        final_dfs = list()
        for exp in experiments:
            exp_path = os.path.join(self.exp_path, exp)
            result = self.read_exp_folder(exp_path)
            n = result.values.shape[0]
            result.insert(3, 'generator', [exp] * n)
            final_dfs.append(result)
            del result
        return pd.concat(final_dfs, axis=0, ignore_index=True)

    def read_exp_folder(self, folder):
        datasets_path = os.path.join(self.exp_path, folder)
        datasets = self.list_dir(datasets_path)
        f1_metrics = list()
        roc_metrics = list()
        for ds in datasets:
            ds_path = os.path.join(self.exp_path, folder, ds)
            f1, roc_auc = self.read_ds_data(ds_path)
            f1_metrics.append(f1)
            roc_metrics.append(roc_auc)

        return pd.DataFrame.from_dict(dict(dataset=datasets,
                                           f1=f1_metrics,
                                           roc_auc=roc_metrics))

    def read_ds_data(self, path):
        launches = self.list_dir(path)
        f1_list = list()
        roc_list = list()
        for launch in launches:
            single_exp_path = os.path.join(path, launch, 'test_results', 'metrics.csv')
            try:
                f1, roc = pd.read_csv(single_exp_path, index_col=0)['1'][:2]
            except KeyError:
                f1 = pd.read_csv(single_exp_path, index_col=0).loc[0, 'f1']
                roc = pd.read_csv(single_exp_path, index_col=0).loc[0, 'roc_auc']
            f1_list.append(f1)
            roc_list.append(roc)
        return np.mean(f1_list), np.mean(roc_list)

    @staticmethod
    def list_dir(path):
        """Function used instead of ``os.listdir()`` to get list of non-hidden directories.

        Args:
            path (str): Path to the directory.

        Returns:
            list: List of non-hidden directories.

        """
        path_list = []
        for f in os.listdir(path):
            if '.' not in f:
                path_list.append(f)
        return path_list


# Example of usage:
if __name__ == '__main__':
    # mega_table_path = '/results_of_experiments/MegaComparisonResultsSheet.xls'
    # mega_table_f1 = pd.read_excel(mega_table_path, sheet_name='F1Test')
    # mega_table_roc = pd.read_excel(mega_table_path, sheet_name='AUROCTest')

    parser = ResultsParser()
    results_table = parser.run()
    results_table.to_csv('results_4.csv', index=False)
