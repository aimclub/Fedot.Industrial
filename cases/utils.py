import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import RegressionMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, \
    explained_variance_score, max_error, d2_absolute_error_score

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


def calculate_regression_metric(test_target, labels):
    test_target = test_target.astype(np.float)
    metric_dict = {'r2_score:': r2_score(test_target, labels),
                   'mean_squared_error:': mean_squared_error(test_target, labels),
                   'root_mean_squared_error:': np.sqrt(mean_squared_error(test_target, labels)),
                   'mean_absolute_error': mean_absolute_error(test_target, labels),
                   'median_absolute_error': median_absolute_error(test_target, labels),
                   'explained_variance_score': explained_variance_score(test_target, labels),
                   'max_error': max_error(test_target, labels),
                   'd2_absolute_error_score': d2_absolute_error_score(test_target, labels)
                   # 'root_mean_squared_log_error': mean_squared_log_error(test_target, labels, squared=False)
                   }
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    return df


def evaluate_industrial_model(train_data, test_data, task: str = 'regression'):
    metric_dict = {}
    input_data = init_input_data(train_data[0], train_data[1], task=task)
    val_data = init_input_data(test_data[0], test_data[1], task=task)
    for model in model_dict.keys():
        print(f'Current_model - {model}')
        pipeline = model_dict[model].build()
        pipeline.fit(input_data)
        features = pipeline.predict(val_data).predict
        metric = calculate_regression_metric(test_target=test_data[1], labels=features)
        metric_dict.update({model: metric})
    return metric_dict


def tuning_industrial_pipelines(pipeline, tuning_params, train_data):
    input_data = init_input_data(train_data[0], train_data[1], task=tuning_params['task'])
    tuning_method = SimultaneousTuner
    pipeline_tuner = TunerBuilder(input_data.task) \
        .with_tuner(tuning_method) \
        .with_metric(tuning_params['metric']) \
        .with_timeout(tuning_params['tuning_timeout']) \
        .with_iterations(tuning_params['tuning_iterations']) \
        .build(input_data)

    pipeline = pipeline_tuner.tune(pipeline)
    return pipeline


def evaluate_automl(runs=5):
    metric_dict = {}
    model_list = []

    if 'tuning_params' in experiment_setup.keys():
        del experiment_setup['tuning_params']

    if 'industrial_preprocessing' in experiment_setup.keys():
        ind_preproc = experiment_setup['industrial_preprocessing']
        del experiment_setup['industrial_preprocessing']
    else:
        ind_preproc = True

    for run in range(runs):
        model = FedotIndustrial(**experiment_setup)
        model.preprocessing = ind_preproc
        model.fit(train_data)
        prediction = model.predict(test_data)

        metric = calculate_regression_metric(test_target=test_data[1], labels=prediction)
        metric = metric.T
        metric.columns = metric.columns.values

        metric_dict.update({f'run_number - {run}': metric})
        model_list.append(model)

    return metric_dict, model_list


def finetune(tuning_params, model_dict, train_data, test_data, val_data, input_data):
    metric_dict = {}
    for model in model_dict.keys():
        print(f'Current_model - {model}')
        pipeline = model_dict[model].build()
        tuned_pipeline = tuning_industrial_pipelines(pipeline, tuning_params, train_data)
        tuned_pipeline.fit(input_data)
        features = tuned_pipeline.predict(val_data).predict
        metric = calculate_regression_metric(test_target=test_data[1], labels=features)
        metric = metric.T
        metric.columns = metric.columns.values
        metric['model_params'] = metric['model_params'] = str(
            {node: node.parameters for node in tuned_pipeline.graph_description['nodes']})
        metric_dict.update({model: metric})
    return metric_dict


def ts_regression_setup():
    model_dict = {
        'regression_with_statistical_features': PipelineBuilder().add_node('quantile_extractor',
                                                                           params={'window_size': 0}).add_node('ridge')
    }
    experiment_setup = {'problem': 'regression',
                        'metric': 'rmse',
                        'timeout': 20,
                        'num_of_generations': 5,
                        'pop_size': 10,
                        'logging_level': 40,
                        'available_operations':
                            ['rfr',
                             'ridge',
                             'gbr',
                             'sgdr',
                             'linear',
                             'xgbreg',
                             'dtreg',
                             'treg',
                             'scaling',
                             'normalization',
                             'kernel_pca',
                             'eigen_basis',
                             'fourier_basis',
                             'minirocket_extractor',
                             'quantile_extractor',
                             'signal_extractor'
                             ],
                        'n_jobs': 4,
                        'industrial_preprocessing': True,
                        'initial_assumption': None,
                        'max_pipeline_fit_time': 5,
                        'with_tuning': False,
                        'early_stopping_iterations': 5,
                        'early_stopping_timeout': 10,
                        'optimizer': IndustrialEvoOptimizer}

    OperationTypesRepository = IndustrialModels().setup_repository()
    tuning_params = {'task': 'regression',
                     'metric': RegressionMetricsEnum.RMSE,
                     'tuning_timeout': 10,
                     'tuning_iterations': 30}
    data_path = PROJECT_PATH + '/examples/data'
    return OperationTypesRepository, tuning_params, data_path, experiment_setup, model_dict


def sota_compare(data_path, dataset_name, best_baseline, best_tuned, df_automl):
    df = pd.read_csv(data_path + '/ts_regression_sota_results.csv', sep=';')
    df = df[df['ds/type'] == dataset_name].iloc[:, :25]
    df.index = df['algorithm']
    df = df.drop(['algorithm', 'ds/type'], axis=1)
    df = df.replace(',', '.', regex=True).astype(float)
    df['Fedot_Industrial_baseline'] = best_baseline
    df['Fedot_Industrial_tuned'] = best_tuned
    df['Fedot_Industrial_AutoML'] = 0
    df.loc['min', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].min()
    df.loc['max', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].max()
    df.loc['average', 'Fedot_Industrial_AutoML'] = df_automl['root_mean_squared_error:'].mean()
    df = df.T
    df.sort_values('min')['min']
    df.sort_values('max')['max']
    df.sort_values('average')['average']
