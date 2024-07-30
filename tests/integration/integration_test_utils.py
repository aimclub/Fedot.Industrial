from fedot_ind.api.main import FedotIndustrial

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate

METRICS = {
    'classification': 'f1',
    'regression': 'mse',
    'ts_forecasting': 'mse'
}
FINETUNE = False

def data(name):
    train_data, test_data = DataLoader(dataset_name=name).load_data()
    return train_data, test_data


def basic_launch(task, train_data, test_data):
    industrial = FedotIndustrial(problem=task,
                                 timeout=0.1,
                                 n_jobs=-1,
                                 )
    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    return labels, probs


def launch_api(problem, industrial_strategy, train_data, test_data, **other_configs):
    api_config = dict(problem=problem,
                  metric=METRICS[problem],
                  timeout=0.1,
                  n_jobs=-1,
                  industrial_strategy=industrial_strategy,
                  industrial_task_params={'industrial_task': problem,
                                          'data_type': 'time_series'},
                  use_input_preprocessing=True,
                  industrial_strategy_params={},
                  logging_level=20) | other_configs

    industrial = FedotIndustrial(**api_config)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    probs = industrial.predict_proba(test_data)
    assert labels is not None
    assert probs is not None
    return labels, probs

def launch_api(problem, industrial_strategy, dataset_name, **other_configs):
    api_config = dict(problem=problem,
                  metric=METRICS[problem],
                  timeout=0.1,
                  n_jobs=-1,
                  industrial_strategy=industrial_strategy,
                  industrial_task_params={'industrial_task': problem,
                                          'data_type': 'time_series'},
                  use_input_preprocessing=True,
                  industrial_strategy_params={},
                  logging_level=20) | other_configs
    result_dict = ApiTemplate(api_config=api_config,
                                  metric_list=METRICS[problem]
                                  ).eval(dataset=dataset_name, finetune=FINETUNE)
    assert result_dict is not None
