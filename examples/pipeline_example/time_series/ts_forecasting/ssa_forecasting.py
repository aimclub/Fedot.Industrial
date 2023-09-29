import numpy as np
from matplotlib import pyplot as plt

from fedot.core.composer.metrics import smape
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from examples.example_utils import get_ts_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.api.utils.path_lib import PROJECT_PATH


def plot_metrics_and_prediction(test_data,
                                train_data,
                                model_prediction,
                                baseline_prediction,
                                model_name,
                                dataset_name):
    plt.title(dataset_name)
    plt.plot(train_data.idx, test_data.features, label='features')
    plt.plot(test_data.idx, test_data.target, label='target')
    plt.plot(test_data.idx, model_prediction, label=model_name)
    plt.plot(test_data.idx, baseline_prediction, label='predicted baseline')
    plt.grid()
    plt.legend()
    plt.show()

    print(f"SSA smape: {smape(test_data.target, model_prediction)}")
    print(f"no SSA smape: {smape(test_data.target, baseline_prediction)}")


model_dict = {'ssa_forecasting': PipelineBuilder().add_node('data_driven_basis_for_forecasting',
                                                            params={'window_size': 35}
                                                            ),
              'baseline': PipelineBuilder().add_node('ar')}

datasets = {
    'm4_yearly': PROJECT_PATH + f'/examples/data/ts/M4YearlyTest.csv',
    'm4_weekly': PROJECT_PATH + f'/examples/data/ts/M4WeeklyTest.csv',
    'm4_daily': PROJECT_PATH + f'/examples/data/ts/M4DailyTest.csv',
    'm4_monthly': PROJECT_PATH + f'/examples/data/ts/M4MonthlyTest.csv',
    'm4_quarterly': PROJECT_PATH + f'/examples/data/ts/M4QuarterlyTest.csv'}

forecast_length = 13

if __name__ == '__main__':

    train_data, test_data, dataset_name = get_ts_data(datasets, 'm4_monthly', forecast_length)
    baseline = model_dict['baseline'].build()
    del model_dict['baseline']
    baseline.fit(train_data)
    baseline_prediction = np.ravel(baseline.predict(test_data).predict)
    with IndustrialModels():
        for model in model_dict.keys():
            pipeline = model_dict[model].build()
            pipeline.fit(train_data)
            model_prediction = np.ravel(pipeline.predict(test_data).predict)
            plot_metrics_and_prediction(test_data,
                                        train_data,
                                        model_prediction,
                                        baseline_prediction,
                                        model,
                                        dataset_name)
