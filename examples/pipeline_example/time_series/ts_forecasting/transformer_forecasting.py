import matplotlib.pyplot as plt
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

from examples.example_utils import get_ts_data
from examples.pipeline_example.time_series.ts_forecasting.ssa_forecasting import plot_metrics_and_prediction
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

if __name__ == '__main__':

    forecast_length = 16
    window_size_percentage = 10

    train_data, test_data, dataset_name = get_ts_data('m4_daily',
                                                      forecast_length,
                                                      m4_id='D1101')

    patch_len = WindowSizeSelector(
        method='dff').get_window_size(train_data.features)
    window_length_heuristic = round(
        train_data.features.shape[0] / 100 * window_size_percentage)
    window_length_hac = patch_len * 3
    window_length = max(window_length_hac, window_length_heuristic)
    OperationTypesRepository = IndustrialModels().setup_repository()
    model_dict = {
        'tst_model':
            PipelineBuilder().add_node('patch_tst_model', params={'patch_len': None,
                                                                  'forecast_length': forecast_length,
                                                                  'epochs': 50}),
        'spectral_tst_model':
            PipelineBuilder().add_node('ar').add_node('eigen_basis',
                                                      params={'window_size': window_length}, branch_idx=1).
            add_node('feature_filter_model', params={'grouping_level': 0.5}, branch_idx=1).
            add_node('patch_tst_model', params={'patch_len': None,
                                                'forecast_length': forecast_length,
                                                'epochs': 1}, branch_idx=1),

        'baseline': PipelineBuilder().add_node('ar')
    }
    baseline = model_dict['baseline'].build()
    del model_dict['baseline']
    baseline.fit(train_data)
    baseline_prediction = np.ravel(baseline.predict(test_data).predict)
    error_pipeline = PipelineBuilder().add_node('lagged').add_node('ssa_forecaster').add_node('ts_naive_average',
                                                                                              branch_idx=1).join_branches('lasso').build()
    for model in model_dict.keys():
        pipeline = model_dict[model].build()
        pipeline.fit(train_data)
        model = Fedot(problem='ts_forecasting',
                      logging_level=20,
                      n_jobs=1,
                      preset='ts',
                      available_operations=[
                          'ssa_forecaster',
                          'patch_tst_model',
                          'ar',
                          # 'ridge',
                          'ts_naive_average',
                          'stl_arima',
                          'lagged',
                          'arima',
                          'lasso'
                      ],
                      task_params=TsForecastingParams(
                          forecast_length=forecast_length),
                      timeout=180
                      )
        preds = pipeline.predict(test_data)
        error_pipeline.fit(train_data)
        model.fit(train_data)
        model.current_pipeline.show()
        model_prediction = model.predict(test_data)
        #
        # model_prediction = pipeline.predict(test_data).predict
        plot_metrics_and_prediction(test_data,
                                    train_data,
                                    model_prediction,
                                    baseline_prediction,
                                    model,
                                    dataset_name)
