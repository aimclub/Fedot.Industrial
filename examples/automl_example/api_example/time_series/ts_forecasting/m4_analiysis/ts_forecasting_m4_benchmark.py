import os

import pandas as pd

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH, M4_FORECASTING_LENGTH
from fedot_ind.tools.example_utils import industrial_forecasting_modelling_loop, compare_forecast_with_sota, \
    read_results, create_comprasion_df

forecast_col = ['industrial', 'target', 'AG', 'NBEATS']
metric_col = ['industrial', 'AG', 'NBEATS']
benchmark = 'M4'
finetune = False

forecast_result_path = os.listdir(
    PROJECT_PATH +
    '/examples/automl_example/api_example/time_series/ts_forecasting/forecasts/')
forecast_result_path = set([x.split('_')[0] for x in forecast_result_path])

df_forecast, df_metrics = read_results(
    PROJECT_PATH + '/examples/automl_example/api_example/time_series/ts_forecasting/forecasts/')
df_comprasion = create_comprasion_df(df_metrics, 'rmse')

if __name__ == "__main__":
    industrial_loss = df_comprasion[df_comprasion['industrial_Wins']
                                    == 'Loose']['dataset_name'].values.tolist()

    api_config = dict(problem='ts_forecasting',
                      metric='rmse',
                      timeout=15,
                      with_tuning=False,
                      pop_size=10,
                      industrial_strategy='forecasting_assumptions',
                      n_jobs=2,
                      logging_level=40)

    for dataset_name in M4_FORECASTING_BENCH:
        if dataset_name in industrial_loss and dataset_name.__contains__('W'):
            print('Already evaluated, but with bad metrics')
            horizon = M4_FORECASTING_LENGTH[dataset_name[0]]
            api_config.update(task_params={'forecast_length': horizon})
            api_config.update(output_folder=os.path.join(
                PROJECT_PATH, 'results_of_experiments', dataset_name))
            n_beats_forecast, n_beats_metrics, \
                autogluon_forecast, autogluon_metrics = compare_forecast_with_sota(dataset_name=dataset_name,
                                                                                   horizon=horizon)
            model, labels, metrics, target = industrial_forecasting_modelling_loop(dataset_name=dataset_name,
                                                                                   benchmark=benchmark,
                                                                                   horizon=horizon,
                                                                                   api_config=api_config,
                                                                                   finetune=finetune)

            forecast = pd.DataFrame(
                [labels, target, n_beats_forecast, autogluon_forecast]).T
            forecast.columns = forecast_col

            metrics_comprasion = pd.concat(
                [metrics, autogluon_metrics, n_beats_metrics]).T
            metrics_comprasion.columns = metric_col

            model.save_best_model()
            model.save_optimization_history()

            if metrics_comprasion.T[metrics_comprasion.T['rmse'] == metrics_comprasion.T.min(
                    axis=0).values[0]].index[0] == 'industrial':
                forecast.to_csv(f'./{dataset_name}_forecast.csv')
                metrics_comprasion.to_csv(f'./{dataset_name}_metrics.csv')

        elif dataset_name in forecast_result_path:
            print('Already evaluated')
