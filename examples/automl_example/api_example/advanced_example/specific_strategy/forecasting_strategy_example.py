import pickle

from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH
from fedot_ind.tools.example_utils import industrial_common_modelling_loop

finetune = False


def forecasting_loop(dataset_dict, api_config):
    metric_names = ('rmse', 'smape')
    model, labels, metrics = industrial_common_modelling_loop(
        api_config=api_config, dataset_name=dataset_dict, finetune=finetune, metric_names=metric_names)
    return model, labels, metrics


def evaluate_for_M4(type: str = 'M'):
    dataset_list = [data for data in M4_FORECASTING_BENCH if data.__contains__(type)]
    return dataset_list


if __name__ == "__main__":
    bench = 'M4'
    group = 'M'
    forecast_params = {'forecast_length': 8}
    horizon = forecast_params['forecast_length']
    dataset_list = evaluate_for_M4(group)
    api_config = dict(
        problem='ts_forecasting',
        metric='rmse',
        timeout=1,
        with_tuning=False,
        industrial_strategy='forecasting_assumptions',
        industrial_strategy_params={
            'industrial_task': 'ts_forecasting',
            'data_type': 'time_series'},
        task_params=forecast_params,
        logging_level=50)
    result_dict = {}

    for dataset_name in dataset_list:
        dataset_dict = {'benchmark': bench,
                        'dataset': dataset_name,
                        'task_params': forecast_params}
        model, labels, metrics = forecasting_loop(dataset_dict, api_config)
        result_dict.update({dataset_name: dict(metrics=metrics,
                                               forecast=labels)})

    with open(f'{bench}_{group}_forecast_length_{horizon}.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
