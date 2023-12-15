import pandas as pd
from sklearn.metrics import accuracy_score

from examples.benchmark.time_series_multi_reg_benchmark import evaluate_loop
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tsml_eval._wip.results.results_by_classifier import *

available_operations = [
    'eigen_basis',
    'dimension_reduction',
    'inception_model',
    'logit',
    'rf',
    'xgboost',
    'minirocket_extractor',
    'normalization',
    'omniscale_model',
    'pca',
    'mlp',
    'quantile_extractor',
    # 'resample',
    'scaling',
    'signal_extractor',
    'topological_features'
]
experiment_setup = {'task': 'classification',
                    'metric': 'accuracy',
                    'timeout': 30,
                    'available_operations': available_operations,
                    'n_jobs': 2,
                    'max_pipeline_fit_time': 4,
                    'optimizer': IndustrialEvoOptimizer}

dataset_list = multivariate_equal_length
metric_dict = {}

if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    try:
        results = pd.read_csv('./time_series_multi_reg_comparasion.csv', sep=';', index_col=0)
    except Exception:
        results = get_averaged_results_from_web(datasets=multivariate_equal_length, classifiers=valid_multi_classifiers)
        results = pd.DataFrame(results)
        results.columns = valid_multi_classifiers
        results.index = multivariate_equal_length

    results['Fedot_Ind'] = 0
    for dataset in dataset_list:
        prediction, target = evaluate_loop(dataset, experiment_setup)
        try:
            metric = accuracy_score(y_true=target, y_pred=prediction)
        except Exception:
            metric = accuracy_score(y_true=target, y_pred=np.argmax(prediction, axis=1))
        metric_dict.update({dataset: metric})
        results.loc[dataset, 'Fedot_Ind'] = metric
        results.to_csv('./time_series_multi_clf_industrial_run.csv')
        print(metric_dict)
