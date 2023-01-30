import os
from typing import Union

import pandas as pd

from core.architecture.postprocessing.results_picker import ResultsPicker
from core.ensemble.static.RankEnsembler import RankEnsemble


class RankEnsembleExample:
    def __init__(self, results_path: str = None, launch_type: Union[str, int] = 'max'):
        self.results_path = results_path
        self.launch_type = launch_type
        self.parser = ResultsPicker(launch_type=launch_type,
                                    path=self.results_path, ) # path none -> read from results_of_experiments

    def load_results(self):
        proba_dict, metric_dict = self.parser.get_metrics_and_proba()
        return proba_dict, metric_dict

    def apply_rank_ensemble(self, proba_dict: dict, metric_dict: dict):
        exp_results = dict()
        for dataset in proba_dict:
            print(f'ENSEMBLE FOR DATASET - {dataset}'.center(50, '–'))
            modelling_proba = proba_dict[dataset]
            modelling_metrics = metric_dict[dataset]
            rank_ensemble = RankEnsemble(dataset_name=dataset)

            exp_results.update({dataset: rank_ensemble.ensemble((modelling_proba,
                                                                 modelling_metrics))})
        return exp_results

    def create_report(self, experiment_results: dict):
        experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
        experiment_df = experiment_df.fillna(0)
        if 'Best_ensemble_metric' not in experiment_df.columns:
            experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
        experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
        experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
        return experiment_df

    def run(self, save_report: bool = False):
        proba_dict, metric_dict = self.load_results()
        ensemble_results = self.apply_rank_ensemble(proba_dict=proba_dict,
                                                    metric_dict=metric_dict)
        ensemble_report = self.create_report(experiment_results=ensemble_results)
        if save_report:
            ensemble_report.to_csv(os.path.join(self.results_path, 'experiment_report.csv'))
        return ensemble_report


if __name__ == '__main__':

    solver = RankEnsembleExample(launch_type='max', results_path=None)
    report = solver.run(save_report=False)
    print(report)

