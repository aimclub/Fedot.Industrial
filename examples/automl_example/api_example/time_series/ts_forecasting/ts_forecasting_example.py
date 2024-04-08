import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.metrics.metrics_implementation import calculate_forecasting_metric
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH
from fedot_ind.tools.loader import DataLoader

if __name__ == "__main__":

    #dataset_name = 'D1317'
    benchmark = 'M4'
    horizon = 14
    finetune = False
    for dataset_name in M4_FORECASTING_BENCH:
        try:
            autogluon = PROJECT_PATH + f'/benchmark/results/benchmark_results/autogluon/' \
                                       f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
            n_beats = PROJECT_PATH + f'/benchmark/results/benchmark_results/nbeats/' \
                                     f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
            n_beats = pd.read_csv(n_beats)
            autogluon = pd.read_csv(autogluon)

            n_beats_forecast = calculate_forecasting_metric(target=n_beats['value'].values,
                                                            labels=n_beats['predict'].values)
            autogluon_forecast = calculate_forecasting_metric(target=autogluon['value'].values,
                                                              labels=autogluon['predict'].values)

            initial_assumption = PipelineBuilder().add_node('eigen_basis',
                                                            params={'low_rank_approximation': False,
                                                                    'rank_regularization': 'explained_dispersion'}).add_node(
                'ar')
            industrial = FedotIndustrial(problem='ts_forecasting',
                                         metric='rmse',
                                         task_params={'forecast_length': horizon},
                                         timeout=5,
                                         with_tuning=False,
                                         initial_assumption=initial_assumption,
                                         n_jobs=2,
                                         logging_level=30)

            train_data, _ = DataLoader(dataset_name=dataset_name).load_forecast_data(folder=benchmark)

            if finetune:
                model = industrial.finetune(train_data)
            else:
                model = industrial.fit(train_data)

            labels = industrial.predict(train_data)
            metrics = industrial.get_metrics(target=train_data.values[-horizon:].flatten(),
                                             metric_names=('smape', 'rmse', 'median_absolute_error'))
            industrial.save_best_model()
            forecast = pd.DataFrame([labels,
                                     train_data.values[-horizon:].flatten(),
                                     autogluon['predict'].values,
                                     n_beats['predict'].values]).T
            forecast.columns = ['industrial', 'target',
                                'AG',
                                'NBEATS']
            metrics_comprasion = pd.concat([metrics, autogluon_forecast, n_beats_forecast]).T
            metrics_comprasion.columns = ['industrial',
                                          'AG',
                                          'NBEATS']
            forecast.to_csv(f'./{dataset_name}_forecast.csv')
            metrics_comprasion.to_csv(f'./{dataset_name}_metrics.csv')
        except Exception:
            print(f'Skip {dataset_name}')

