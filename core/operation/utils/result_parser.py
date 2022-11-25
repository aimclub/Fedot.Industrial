import os

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
        self.ds_info_path = os.path.join(PROJECT_PATH, 'core/operation/utils/ds_info.plk')

    def run(self) -> pd.DataFrame:
        """
        Base method for parsing results of experiments.

        Returns:
            Table with results of experiments.

        """
        ds_info = pd.read_pickle(self.ds_info_path)
        exp_results = self.retrieve_data()
        ls = []
        for ds in exp_results['dataset']:
            ls.append(int(ds_info[ds_info['dataset'] == ds]['n_classes']))
        exp_results['n_classes'] = ls

        return exp_results

    def retrieve_data(self):
        experiments = self.list_dir(self.exp_path)
        final_dfs = list()
        for exp in experiments:
            exp_path = os.path.join(self.exp_path, exp)
            result = self.read_exp_folder(exp_path)
            n = result.values.shape[0]
            result.insert(3, 'generator', [exp]*n)
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
            f1, roc = pd.read_csv(single_exp_path, index_col=0)['1'][:2]
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
