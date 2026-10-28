"""Packaged real TRAIN data; all search folds stay inside TRAIN."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass
class SearchData:
    folds: tuple
    refit: object
    test: object
    source: dict


def array_digest(*arrays):
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, str(value.dtype))).encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_data(task, seed, horizon=12, *, include_test=True):
    import torch
    from fedot import create_data
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from sktime.datasets import load_acsf1, load_airline, load_tecator
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    if task == 'ts_forecasting':
        values = load_airline().to_numpy(dtype=float)
        history, future = values[:-horizon], values[-horizon:]
        native_task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))

        def forecast_pair(prefix, validation):
            from fedot.core.data.split.temporal import temporal_tensor_holdout
            joined = np.concatenate((prefix, validation))
            complete = create_data(joined, target=joined.copy(), task=native_task,
                                   ts_forecast_horizon=horizon, without_target=True)
            return temporal_tensor_holdout(complete)

        folds = tuple(forecast_pair(history[:end], history[end:end + horizon])
                      for end in range(len(history) - 3 * horizon, len(history), horizon))
        refit, test = forecast_pair(history, future) if include_test else (folds[-1][0], None)
        if not include_test:
            refit = create_data(history, target=history.copy(), task=native_task,
                                ts_forecast_horizon=horizon, without_target=True)
        source = {'dataset': 'Airline', 'loader': 'sktime.datasets.load_airline',
                  'train_shape': list(history.shape), 'test_shape': list(future.shape),
                  'folds': [{'train_end': len(history) - i * horizon, 'horizon': horizon} for i in (3, 2, 1)],
                  'train_sha256': array_digest(history), 'test_used_for_search': False}
        return SearchData(folds, refit, test, source)

    loader = load_acsf1 if task == 'classification' else load_tecator
    x, y = loader(split='train', return_type='numpy3d')
    xt, yt = loader(split='test', return_type='numpy3d') if include_test else (None, None)
    if task == 'classification':
        encoder = LabelEncoder().fit(y)
        y = encoder.transform(y)
        yt = encoder.transform(yt) if include_test else None
        splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    else:
        y = np.asarray(y, dtype=float)
        yt = np.asarray(yt, dtype=float) if include_test else None
        splitter = KFold(3, shuffle=True, random_state=seed)
    x = np.asarray(x, dtype=np.float32)
    xt = np.asarray(xt, dtype=np.float32) if include_test else None
    native_task = Task(TaskTypesEnum[task])

    def container(features, target):
        return TensorData(features=torch.as_tensor(features), target=torch.as_tensor(target),
                          idx=np.arange(len(target)), task=native_task,
                          data_type=DataTypesEnum.multi_ts)

    folds, records = [], []
    for train_idx, validation_idx in splitter.split(x, y):
        # Each fold owns its preprocessing, fitted without validation or TEST.
        mean = x[train_idx].mean(axis=(0, 2), keepdims=True)
        scale = x[train_idx].std(axis=(0, 2), keepdims=True)
        scale = np.where(scale > 0, scale, 1.)
        folds.append((container((x[train_idx] - mean) / scale, y[train_idx]),
                      container((x[validation_idx] - mean) / scale, y[validation_idx])))
        records.append({'train_indices': train_idx.tolist(), 'validation_indices': validation_idx.tolist()})
    mean, scale = x.mean(axis=(0, 2), keepdims=True), x.std(axis=(0, 2), keepdims=True)
    scale = np.where(scale > 0, scale, 1.)
    return SearchData(tuple(folds), container((x - mean) / scale, y),
                      container((xt - mean) / scale, yt) if include_test else None,
                      {'dataset': 'ACSF1' if task == 'classification' else 'Tecator',
                       'loader': loader.__module__ + '.' + loader.__name__, 'folds': records,
                       'train_shape': list(x.shape), 'test_shape': list(xt.shape) if include_test else None,
                       'train_sha256': array_digest(x, y), 'test_used_for_search': False})
