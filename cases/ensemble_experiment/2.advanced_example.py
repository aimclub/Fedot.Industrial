import os

import pandas as pd
import collections

from core.api.API import Industrial
from core.operation.utils.result_parser import ResultsParser

if __name__ == '__main__':
    # modelling_path = os.listdir('supplementary_data/ensemble_predictions')
    # modelling_results = [os.path.join('supplementary_data/ensemble_predictions', path, 'probs_preds_target.csv') for
    #                      path in
    #                      modelling_path]
    # metrics_results = [os.path.join('supplementary_data/ensemble_predictions', path, 'metrics.csv') for path in
    #                    modelling_path]
    # for result, model in zip(metrics_results, modelling_path):
    #     print(f'*---------BASE RESULT FOR MODEL - {model}')
    #     _ = pd.read_csv(result, index_col=0)
    #     print(_)
    exp_folders = [
        'quantile',
                   #'window_spectral',
                   #'wavelet',
                   'recurrence',
                   'spectral',
                   'window_quantile'
                   ]
    parser = ResultsParser()
    proba_dict, metric_dict = parser.read_proba(path='18.11.22', launch='max', exp_folders=exp_folders)
    experiment_results = {}
    for dataset in proba_dict:
        print(f'*---------ENSEMBLING FOR DATASET - {dataset}')
        modelling_results = proba_dict[dataset]
        modelling_metrics = metric_dict[dataset]
        model_rank = {}

        for model in modelling_metrics:
            print(f'*---------BASE RESULT FOR MODEL - {model}')
            if len(modelling_results[model].columns) == 3:
                metric = 'roc_auc'
                type = 'binary'
            else:
                metric = 'f1'
                type = 'multiclass'
            print(f'*---------TYPE OF ML TASK - {type}. Metric - {metric}')
            print(modelling_metrics[model])
            model_rank.update({model: modelling_metrics[model][metric][0]})

        sorted_dict = dict(sorted(model_rank.items(), key=lambda x: x[1], reverse=True))

        n_models = len(sorted_dict)
        best_base_model = list(sorted_dict)[0]
        best_metric = sorted_dict[best_base_model]
        top_K_models_dict = {}
        best_ensemble_metric = 0
        top_K_models_dict['Base_model'] = best_base_model
        top_K_models_dict['Base_metric'] = best_metric

        print(f'*---------CURRENT BEST METRIC - {best_metric}. MODEL - {best_base_model}')

        for top_K_models in range(n_models):
            print(f'*---------SELECT TOP {top_K_models + 1} MODELS AND APPLY ENSEMBLE.')
            modelling_results_top = {k: v for k, v in modelling_results.items() if
                                     k in list(sorted_dict.keys())[:top_K_models + 1]}
            IndustrialModel = Industrial()
            ensemble_results = IndustrialModel.apply_ensemble(modelling_results=modelling_results_top)
            top_ensemble_dict = {}
            for ensemble_method in ensemble_results:
                ensemble_dict = ensemble_results[ensemble_method]
                ensemble_metrics = IndustrialModel.get_metrics(target=ensemble_dict['target'],
                                                               prediction_label=ensemble_dict['label'],
                                                               prediction_proba=ensemble_dict['proba'])
                print(f'*---------ENSEMBLE RESULT FOR MODEL - {ensemble_method}')
                print(ensemble_metrics['metrics'])
                print(f'*--------------------------------------*')
                ensemble_metric = ensemble_metrics['metrics'][metric]
                if ensemble_metric > best_metric and ensemble_metric > best_ensemble_metric:
                    best_ensemble_metric = ensemble_metric
                    top_ensemble_dict.update({ensemble_method: ensemble_metric})

            if len(top_ensemble_dict) == 0:
                print(f'*---------ENSEMBLE DOESNT IMPROVE RESULTS')
            else:
                top_ensemble_dict = dict(sorted(top_ensemble_dict.items(), key=lambda x: x[1], reverse=True))
                current_ensemble_method = list(top_ensemble_dict)[0]
                best_ensemble_metric = top_ensemble_dict[current_ensemble_method]
                model_combination = list(modelling_results_top)[:top_K_models + 1]
                print(
                    f'*---------ENSEMBLE IMPROVE RESULTS.'
                    f'NEW BEST METRIC - {best_ensemble_metric}. METHOD - {current_ensemble_method}')

        if best_ensemble_metric > 0:
            top_K_models_dict[
                'Ensemble_models'] = f'Models: {model_combination}. Ensemble_method: {current_ensemble_method}'
            top_K_models_dict['Best_ensemble_metric'] = best_ensemble_metric

        experiment_results.update({dataset: top_K_models_dict})

    experiment_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    experiment_df = experiment_df.fillna(0)
    if 'Best_ensemble_metric' not in experiment_df.columns:
        experiment_df['Best_ensemble_metric'] = experiment_df['Base_metric']
    experiment_df['Ensemble_gain'] = (experiment_df['Best_ensemble_metric'] - experiment_df['Base_metric']) * 100
    experiment_df['Ensemble_gain'] = experiment_df['Ensemble_gain'].apply(lambda x: x if x > 0 else 0)
_ = 1
