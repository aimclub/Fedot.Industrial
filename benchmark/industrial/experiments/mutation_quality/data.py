"""Convert reviewed TRAIN splits to the existing measured objective containers."""

from pathlib import Path

import numpy as np

from .preflight import inspect_scenario, read_source


def normalized_pair(x, labels, left, right):
    left, right = np.asarray(left), np.asarray(right)
    mean = x[left].mean(axis=(0, 2), keepdims=True)
    scale = x[left].std(axis=(0, 2), keepdims=True)
    scale = np.where(scale > 0, scale, 1.)
    return ((x[left] - mean) / scale, labels[left]), ((x[right] - mean) / scale, labels[right])


def load_search_data(scenario, protocol, root):
    import torch
    from fedot import create_data
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    from benchmark.industrial.experiments.evolution_parallelism.data import SearchData, array_digest
    from sklearn.preprocessing import LabelEncoder

    root = Path(root)
    record = inspect_scenario(scenario, root, protocol)
    if record['status'] != 'ready_for_split_review':
        raise ValueError(f'TRAIN preflight failed: {record}')
    x, y, source = read_source(scenario, root, protocol)
    if source != record['source']:
        raise ValueError('Source changed after preflight')
    split = record['split']
    if scenario.task == 'ts_forecasting':
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        horizon = protocol.forecast_horizon

        def pair(left, right):
            joined = np.concatenate((left, right))
            complete = create_data(joined, target=joined.copy(), task='ts_forecasting',
                                   ts_forecast_horizon=horizon, without_target=True)
            return temporal_tensor_holdout(complete)

        folds = tuple(pair(x[:item['train_end']], x[item['validation_start']:item['validation_end']])
                      for item in split['folds'])
        refit, outer = pair(x[:split['search_end']], x[split['outer_start']:split['outer_end']])
    else:
        x = x.astype(np.float32)
        search = split['search_indices']
        labels = LabelEncoder().fit(y[search]).transform(y) if scenario.task == 'classification' else y.astype(float)
        native_task = Task(TaskTypesEnum[scenario.task])

        def container(values, target):
            return TensorData(features=torch.as_tensor(values), target=torch.as_tensor(target),
                              idx=np.arange(len(target)), task=native_task, data_type=DataTypesEnum.multi_ts)

        def pair(left, right):
            a, b = normalized_pair(x, labels, left, right)
            return container(*a), container(*b)

        folds = tuple(pair(item['train_indices'], item['validation_indices']) for item in split['folds'])
        refit, outer = pair(search, split['outer_indices'])
    record.update(dataset=scenario.name, split_seed=protocol.split_seed,
                  train_shape=list(refit.features.shape), train_sha256=array_digest(
                      refit.features.detach().cpu().numpy() if hasattr(refit.features, 'detach') else refit.features),
                  test_used_for_search=False, outer_used_for_search=False,
                  outer_role='control_holdout_inside_official_train')
    return SearchData(folds, refit, outer, record)
