import os

import pandas as pd
from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.architecture.utils.utils import PROJECT_PATH
from fedot_ind.core import RankEnsemble


def create_report(experiment_results: dict):
    experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    experiment_df = experiment_df.fillna(0)
    if 'Best_ensemble_metric' not in experiment_df.columns:
        experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
    experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
    experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
    return experiment_df


def load_results(folder_path: str, launch_type, model_list: list):
    parser = ResultsPicker(path=os.path.join(PROJECT_PATH, 'tests', 'data', 'ensemble'))
    proba_dict, metric_dict = parser.get_metrics_and_proba()
    return proba_dict, metric_dict


def apply_rank_ensemble(proba_dict: dict,
                        metric_dict: dict):
    experiment_results = {}
    for dataset in proba_dict:
        print(f'*---------ENSEMBLING FOR DATASET - {dataset}')
        modelling_results = proba_dict[dataset]
        modelling_metrics = metric_dict[dataset]
        rank_ensemble = RankEnsemble(proba_dict=modelling_results,
                                     metric_dict=modelling_metrics)

        experiment_results.update({dataset: rank_ensemble.ensemble()})
    return experiment_results


def test_rank_ensemble():
    exp_folders = [
        'quantile',
        'window_spectral',
        'recurrence'
    ]
    path = 'ensemble'
    launch_type = '1'
    proba_dict, metric_dict = load_results(folder_path=path, launch_type=launch_type, model_list=exp_folders)
    experiment_results = apply_rank_ensemble(proba_dict, metric_dict)
    report_df = create_report(experiment_results)
    assert report_df['Base_model'].values[0] in exp_folders
    assert report_df['Best_ensemble_metric'].values[0] < 1.0
