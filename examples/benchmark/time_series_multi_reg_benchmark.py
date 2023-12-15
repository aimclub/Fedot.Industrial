import matplotlib
import pandas as pd
from fedot import Fedot

from examples.example_utils import init_input_data
from fedot_ind.core.optimizer.IndustrialEvoOptimizer import IndustrialEvoOptimizer
from fedot_ind.tools.loader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from tsml_eval._wip.results.results_by_classifier import *
from statsmodels.tools.eval_measures import rmse


def evaluate_loop(dataset, experiment_setup: dict = None):
    matplotlib.use('TkAgg')
    train_data, test_data = DataLoader(dataset_name=dataset).load_data()
    input_data = init_input_data(train_data[0], train_data[1], task=experiment_setup['task'])
    val_data = init_input_data(test_data[0], test_data[1], task=experiment_setup['task'])

    model = Fedot(problem=experiment_setup['task'],
                  logging_level=20,
                  n_jobs=experiment_setup['n_jobs'],
                  metric=experiment_setup['metric'],
                  pop_size=20,
                  num_of_generations=20,
                  optimizer=experiment_setup['optimizer'],
                  available_operations=experiment_setup['available_operations'],
                  max_pipeline_fit_time=experiment_setup['max_pipeline_fit_time'],
                  timeout=experiment_setup['timeout'],
                  with_tuning=False
                  )
    model.fit(input_data)
    prediction = model.predict(val_data)
    try:
        model.history.save(f"{dataset}_optimisation_history.json")
        model.current_pipeline.show(save_path=f'./{dataset}_best_model.png')
        model.history.show.operations_animated_bar(save_path=f'./{dataset}_history_animated_bars.gif',
                                                   show_fitness=True, dpi=100)
    except Exception:
        print('No_visualisation')
    return prediction, val_data.target


# Regression equal length no missing problems 
monash_regression = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia",
]

# 19 Regression problems with interpolated missing and truncated unequal
monash_regression_nm_eq = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BeijingPM10Quality-no-missing",
    "BeijingPM25Quality-no-missing",
    "BenzeneConcentration-no-missing",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "HouseholdPowerConsumption1-no-missing",
    "HouseholdPowerConsumption2-no-missing",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia-equal-length",
]

available_operations = ['rfr',
                        'ridge',
                        'gbr',
                        'sgdr',
                        'lgbmreg',
                        'linear',
                        'xgbreg',
                        'dtreg',
                        'treg',
                        'knnreg',
                        'scaling',
                        'normalization',
                        'pca',
                        'kernel_pca',
                        'isolation_forest_reg',
                        'eigen_basis',
                        'fourier_basis',
                        #'dimension_reduction',
                        'minirocket_extractor',
                        'quantile_extractor',
                        'signal_extractor',
                        'topological_features'
                        ]

experiment_setup = {'task': 'regression',
                    'metric': 'rmse',
                    'timeout': 30,
                    'available_operations': available_operations,
                    'n_jobs': 2,
                    'max_pipeline_fit_time': 4,
                    'optimizer': IndustrialEvoOptimizer}

dataset_list = monash_regression
metric_dict = {}

if __name__ == "__main__":
    OperationTypesRepository = IndustrialModels().setup_repository()
    results = pd.read_csv('./time_series_multi_reg_comparasion.csv', sep=';', index_col=0)
    results = results.dropna(axis=1, how='all')
    results = results.dropna(axis=0, how='all')
    results['Fedot_Ind'] = 0
    for dataset in dataset_list:
        prediction, target = evaluate_loop(dataset, experiment_setup)
        metric = rmse(target, prediction)[0]
        metric_dict.update({dataset: metric})
        results.loc[dataset, 'Fedot_Ind'] = metric
        results.to_csv('./time_series_multi_reg_industrial_run.csv')
        print(metric_dict)
    _ = 1
