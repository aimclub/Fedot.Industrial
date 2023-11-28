import numpy as np
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from examples.example_utils import get_ts_data
from examples.pipeline_example.time_series.ts_forecasting.ssa_forecasting import plot_metrics_and_prediction
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

if __name__ == '__main__':

    forecast_length = 14
    window_size_percentage = 10
    train_data, test_data, dataset_name = get_ts_data('m4_daily',
                                                      forecast_length,
                                                      m4_id='D1101')
    patch_len = WindowSizeSelector(method='dff').get_window_size(train_data.features)
    window_length_heuristic = round(train_data.features.shape[0] / 100 * window_size_percentage)
    window_length_hac = patch_len * 3
    window_length = max(window_length_hac, window_length_heuristic)
    # window_length = patch_len
    model_dict = {
        # 'tst_model':
        #     PipelineBuilder().add_node('patch_tst_model', params={'patch_len': None,
        #                                                           'forecast_length': forecast_length,
        #                                                           'epochs': 200}),
        'spectral_tst_model':
            PipelineBuilder().add_node('eigen_basis',
                                       params={'window_size': window_length}).
            add_node('feature_filter_model', params={
                'grouping_level': 0.5}).
            add_node('patch_tst_model', params={'patch_len': None,
                                                'forecast_length': forecast_length,
                                                'epochs': 200}),
        'baseline': PipelineBuilder().add_node('ar')}
    baseline = model_dict['baseline'].build()
    del model_dict['baseline']
    baseline.fit(train_data)
    baseline_prediction = np.ravel(baseline.predict(test_data).predict)

    with IndustrialModels():
        for model in model_dict.keys():
            pipeline = model_dict[model].build()
            pipeline.fit(train_data)
            model_prediction = pipeline.predict(test_data).predict
            plot_metrics_and_prediction(test_data,
                                        train_data,
                                        np.sum(model_prediction, axis=0),
                                        baseline_prediction,
                                        model,
                                        dataset_name)
            for pred in model_prediction:
                plot_metrics_and_prediction(test_data,
                                            train_data,
                                            pred,
                                            baseline_prediction,
                                            model,
                                            dataset_name)
            _ = 1
