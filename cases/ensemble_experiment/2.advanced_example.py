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

    parser = ResultsParser()
    proba_dict, metric_dict = parser.read_proba()

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
        sorted_dict = dict(sorted(model_rank.items(), key=lambda x: x[1],reverse=True))
        modelling_results_top = {k: v for k, v in modelling_results.items() if k in list(sorted_dict.keys())[:2]}
        IndustrialModel = Industrial()
        ensemble_results = IndustrialModel.apply_ensemble(modelling_results=modelling_results_top)
        for ensemble_method in ensemble_results:
            ensemble_dict = ensemble_results[ensemble_method]
            ensemble_metrics = IndustrialModel.get_metrics(target=ensemble_dict['target'],
                                                           prediction_label=ensemble_dict['label'],
                                                           prediction_proba=ensemble_dict['proba'])
            print(f'*---------ENSEMBLE RESULT FOR MODEL - {ensemble_method}')
            print(ensemble_metrics['metrics'])
            print(f'*--------------------------------------*')
_ = 1
