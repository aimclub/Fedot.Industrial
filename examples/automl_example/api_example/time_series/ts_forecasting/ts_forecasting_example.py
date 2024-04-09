import os

import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH
from fedot_ind.tools.example_utils import industrial_forecasting_modelling_loop, compare_forecast_with_sota

if __name__ == "__main__":
    forecast_result_path = os.listdir(PROJECT_PATH +
                                      '/examples/automl_example/api_example/time_series/ts_forecasting/forecasts/')
    forecast_result_path = set([x.split('_')[0] for x in forecast_result_path])
    forecast_col = ['industrial', 'target', 'AG', 'NBEATS']
    metric_col = ['industrial', 'AG', 'NBEATS']
    benchmark = 'M4'
    horizon = 14
    finetune = False
    initial_assumption = PipelineBuilder().add_node('eigen_basis',
                                                    params={'low_rank_approximation': False,
                                                            'rank_regularization': 'explained_dispersion'}).add_node(
        'ar')
    api_config = dict(problem='ts_forecasting',
                      metric='rmse',
                      task_params={'forecast_length': horizon},
                      timeout=5,
                      with_tuning=False,
                      initial_assumption=initial_assumption,
                      n_jobs=2,
                      logging_level=30)

    for dataset_name in M4_FORECASTING_BENCH:
        if dataset_name in forecast_result_path:
            print('Already evaluated')
        else:
            try:
                n_beats_forecast, n_beats_metrics, \
                autogluon_forecast, autogluon_metrics = compare_forecast_with_sota(dataset_name=dataset_name,
                                                                                   horizon=horizon)
                model, labels, metrics, target = industrial_forecasting_modelling_loop(dataset_name=dataset_name,
                                                                                       benchmark=benchmark,
                                                                                       horizon=horizon,
                                                                                       api_config=api_config,
                                                                                       finetune=finetune)

                forecast = pd.DataFrame([labels,
                                         target,
                                         n_beats_forecast,
                                         autogluon_forecast]).T
                forecast.columns = forecast_col

                metrics_comprasion = pd.concat([metrics,
                                                autogluon_forecast,
                                                n_beats_forecast]).T
                metrics_comprasion.columns = metric_col

                forecast.to_csv(f'./{dataset_name}_forecast.csv')
                metrics_comprasion.to_csv(f'./{dataset_name}_metrics.csv')

            except Exception as ex:
                print(f'Skip {dataset_name}. Reason - {ex}')
