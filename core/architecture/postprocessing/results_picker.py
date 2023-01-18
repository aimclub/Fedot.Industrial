import os
from typing import Union

import pandas as pd

from core.architecture.utils.utils import PROJECT_PATH


class ResultsPicker:
    """Class for parsing results of experiments. It parses all experiments in ``results_of_experiments``
    folder or user-specified folder and creates dataframe table that availible for further analysis.

    Args:
        path (str): path to folder with experiments. Default is ``None`` and it means that path
                    is ``results_of_experiments``.
        launch_type (str or int): number of launch to be extracted. Default is ``max`` and it means that the best launch
                                  will be extracted.

    Examples:
        >>> from core.architecture.postprocessing.results_picker import ResultsPicker
        >>> collector = ResultsPicker(path='to_your_results_folder', launch_type='max')
        >>> metrics_df = parser.run(get_metrics_df=True)
        >>> metrics_df.to_csv('metrics.csv')

    """

    def __init__(self, path: str = None, launch_type: Union[str, int] = 'max'):
        self.exp_path = self.__get_results_path(path)
        self.launch_type = launch_type
        self._metrics = ('f1', 'roc_auc')
        self.exp_folders = ['window_spectral',
                            'topological',
                            'recurrence',
                            'ensemble',
                            'quantile',
                            'spectral',
                            'wavelet',
                            'window_quantile']
        self.ds_info_path = os.path.join(PROJECT_PATH, 'core/operation/utils/ds_info.csv')

    def __get_results_path(self, path):
        if path:
            return path
        else:
            return os.path.join(PROJECT_PATH, 'results_of_experiments')

    def run(self, get_metrics_df: bool = False):
        """
        Base method for parsing results of experiments.

        Returns:
            Table with results of experiments.

        """

        proba_dict, metric_dict = self.get_metrics_and_proba()

        if get_metrics_df:
            return self._create_metrics_df(metric_dict)

        return proba_dict, metric_dict

    def _create_metrics_df(self, metric_dict):
        columns = ['dataset', 'experiment']
        metrics_df = pd.DataFrame()
        for ds in metric_dict.keys():
            for exp in metric_dict[ds].keys():
                metrics = metric_dict[ds][exp]
                metrics_df = metrics_df.append({'dataset': ds, 'experiment': exp, 'f1': metrics['f1'][0],
                                                'roc_auc': metrics['roc_auc'][0], 'accuracy': metrics['accuracy'][0],
                                                'precision': metrics['precision'][0], 'logloss': metrics['logloss'][0]},
                                               ignore_index=True)

        metrics_df = pd.concat([metrics_df[['dataset', 'experiment']],
                                metrics_df[[col for col in metrics_df.columns if col not in columns]]], axis=1)
        return metrics_df

    def get_metrics_and_proba(self):
        experiments = self.list_dir(self.exp_path)
        proba_dict = {}
        metric_dict = {}
        for exp in experiments:
            exp_path = os.path.join(self.exp_path, exp)
            ds_list, metrics_list, proba_list = self.read_exp_folder(exp_path)

            for metric, proba, dataset in zip(metrics_list, proba_list, ds_list):
                if dataset not in proba_dict.keys():
                    proba_dict[dataset] = {}
                if dataset not in metric_dict.keys():
                    metric_dict[dataset] = {}
                proba_dict[dataset].update({exp: proba})
                metric_dict[dataset].update({exp: metric})

        return proba_dict, metric_dict

    def read_exp_folder(self, folder):
        datasets_path = os.path.join(self.exp_path, folder)
        datasets = self.list_dir(datasets_path)
        metrics_list = []
        proba_list = []
        for ds in datasets:
            ds_path = os.path.join(self.exp_path, folder, ds)
            metrics, proba = self.read_ds_data(ds_path)
            metrics_list.append(metrics)
            proba_list.append(proba)

        return datasets, metrics_list, proba_list

    def read_ds_data(self, path):
        if self.launch_type == 'max':
            best_launch = self.find_best_launch(path)
        else:
            best_launch = self.launch_type
        metrics_path = os.path.join(path, best_launch, 'test_results', 'metrics.csv')
        proba_path = os.path.join(path, best_launch, 'test_results', 'probs_preds_target.csv')

        proba = pd.read_csv(proba_path, index_col=0)
        metrics = pd.read_csv(metrics_path, index_col=0)
        if 'index' in metrics.columns:
            del metrics['index']
            metrics = metrics.T
            metrics = metrics.rename(columns=metrics.iloc[0])
            metrics = metrics[1:]

        return metrics, proba

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

    def find_best_launch(self, launch_folders):
        best_metric = 0
        launch = 1
        for _dir in self.list_dir(launch_folders):
            metric_path = os.path.join(launch_folders, str(_dir), 'test_results', 'metrics.csv')
            metrics = pd.read_csv(metric_path, index_col=0)
            if 'index' in metrics.columns:
                del metrics['index']
                metrics = metrics.T
                metrics = metrics.rename(columns=metrics.iloc[0])
                metrics = metrics[1:]
            metric_sum = metrics['roc_auc'].values[0] + metrics['f1'].values[0]
            if metric_sum > best_metric:
                best_metric = metric_sum
                launch = _dir
        return launch


# Example of usage:
if __name__ == '__main__':

    parser = ResultsPicker()
    results_table = parser.run(get_metrics_df=True)
    results_table.to_csv('results_4.csv', index=False)
