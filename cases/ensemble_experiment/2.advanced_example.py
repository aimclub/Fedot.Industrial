import pandas as pd
from core.operation.utils.result_parser import ResultsParser
from core.ensemble.static.RankEnsembler import RankEnsemble


def create_report(experiment_results: dict):
    experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    experiment_df = experiment_df.fillna(0)
    if 'Best_ensemble_metric' not in experiment_df.columns:
        experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
    experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
    experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
    return experiment_df


if __name__ == '__main__':

    exp_folders = [
        'quantile',
        # 'window_spectral',
        # 'wavelet',
        'recurrence',
        'spectral',
        'window_quantile'
    ]
    parser = ResultsParser()
    proba_dict, metric_dict = parser.read_proba(path='18.11.22', launch='max', exp_folders=exp_folders)
    experiment_results = {}
    rank_ensemble = RankEnsemble(prediction_proba_dict=proba_dict,
                                 metric_dict=metric_dict)

    for dataset in proba_dict:
        print(f'*---------ENSEMBLING FOR DATASET - {dataset}')
        modelling_results = proba_dict[dataset]
        modelling_metrics = metric_dict[dataset]
        rank_ensemble = RankEnsemble(prediction_proba_dict=modelling_results,
                                     metric_dict=modelling_metrics)

        rank_ensemble.ensemble()

_ = 1
