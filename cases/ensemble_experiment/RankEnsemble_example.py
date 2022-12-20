import pandas as pd
from core.architecture.postprocessing.Parser import ResultsParser
from core.ensemble.static.RankEnsembler import RankEnsemble


def create_report(experiment_results: dict):
    experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    experiment_df = experiment_df.fillna(0)
    if 'Best_ensemble_metric' not in experiment_df.columns:
        experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
    experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
    experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
    return experiment_df


def load_results(folder_path: str, launch_type, model_list: list):
    parser = ResultsParser()
    proba_dict, metric_dict = parser.read_proba(path=folder_path, launch=launch_type, exp_folders=model_list)
    return proba_dict, metric_dict


def apply_rank_ensemble(proba_dict: dict,
                        metric_dict: dict):
    experiment_results = {}
    for dataset in proba_dict:
        print(f'*---------ENSEMBLING FOR DATASET - {dataset}')
        modelling_results = proba_dict[dataset]
        modelling_metrics = metric_dict[dataset]
        rank_ensemble = RankEnsemble(prediction_proba_dict=modelling_results,
                                     metric_dict=modelling_metrics)

        experiment_results.update({dataset: rank_ensemble.ensemble()})
    return experiment_results


if __name__ == '__main__':
    exp_folders = [
        'quantile',
        'window_spectral',
        'wavelet',
        'recurrence',
        'spectral',
        'window_quantile'
    ]
    path = '15.11.22'
    launch_type = 'max'
    proba_dict, metric_dict = load_results(folder_path=path, launch_type=launch_type, model_list=exp_folders)
    experiment_results = apply_rank_ensemble(proba_dict, metric_dict)
    report_df = create_report(experiment_results)
