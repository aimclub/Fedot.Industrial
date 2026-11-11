"""Original benchmark splits; held-out test data never enters search."""

from dataclasses import dataclass
from hashlib import sha256
import csv
from pathlib import Path

import numpy as np


@dataclass
class StudyData:
    train: object
    validation: object
    refit: object
    test: object
    scale: float
    source: dict


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_forecast_series(config, name):
    result = []
    for key in ('forecast_train', 'forecast_test'):
        with Path(config[key]).open(newline='', encoding='utf-8') as source:
            row = next((row for row in csv.reader(source) if row and row[0] == name), None)
        if row is None:
            raise ValueError(f'Missing series {name} in {key}')
        values = np.asarray([float(x) for x in row[1:] if x.strip()], dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f'Invalid forecast series in {key}')
        result.append(values)
    return result


def load_study_data(config, scenario, seed):
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    task, name = scenario['task'], scenario['name']
    if task == 'ts_forecasting':
        from fedot import create_data
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        history, future = load_forecast_series(config, name)
        horizon = len(future)
        full = np.concatenate((history, future))
        data = create_data(full, target=full, task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon)),
                           ts_forecast_horizon=horizon, without_target=True)
        refit, test = temporal_tensor_holdout(data)
        train, validation = temporal_tensor_holdout(refit)
        scale = float(np.mean(np.abs(np.diff(history[:-horizon]))))
        paths = [config['forecast_train'], config['forecast_test']]
        split = {'train_end': len(history) - horizon, 'validation_end': len(history), 'test_end': len(full)}
    else:
        from benchmark.industrial.datasets.local_io import load_local_supervised_split, resolve_local_split_paths
        root = config['classification_root'] if task == 'classification' else config['supervised_root']
        paths = resolve_local_split_paths(name, data_root=root)
        raw = load_local_supervised_split(name, data_root=root)
        x, xt = raw.train_features, raw.test_features
        y, yt = raw.train_target, raw.test_target
        if task == 'classification':
            encoder = LabelEncoder().fit(y)
            y, yt = encoder.transform(y), encoder.transform(yt)
        else:
            y, yt = y.astype(float), yt.astype(float)
        if not all(np.isfinite(v).all() for v in (x, xt, y, yt)):
            raise ValueError('Only complete finite original data is supported by this study')
        indices = np.arange(len(y))
        tr, val = train_test_split(indices, test_size=0.25, random_state=seed,
                                   stratify=y if task == 'classification' else None)

        def tensor(features, target, idx):
            features = features[:, None, :] if features.ndim == 2 else features
            return TensorData(features=torch.as_tensor(features, dtype=torch.float32),
                              target=torch.as_tensor(target), idx=idx, data_type=DataTypesEnum.ts,
                              task=Task(TaskTypesEnum[task]), state='fit')

        train, validation = tensor(x[tr], y[tr], tr), tensor(x[val], y[val], val)
        refit, test = tensor(x, y, indices), tensor(xt, yt, np.arange(len(yt)))
        scale = 1. if task == 'classification' else float(np.std(y[tr]))
        split = {'train_indices': tr.tolist(), 'validation_indices': val.tolist(),
                 'official_train_count': len(y), 'official_test_count': len(yt)}
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('A positive training-only normalization scale is required')
    source = {'files': {str(p): file_hash(p) for p in paths}, 'split': split,
              'normalization_scale': scale, 'dataset': name, 'task': task}
    return StudyData(train, validation, refit, test, scale, source)


def loss_value(task, target, prediction, scale):
    from sklearn.metrics import f1_score, mean_squared_error
    values = np.asarray(prediction)
    target = np.asarray(target).reshape(-1)
    if task == 'classification':
        values = values.argmax(axis=1) if values.ndim == 2 and values.shape[1] > 1 else values.reshape(-1)
        return 1. - float(f1_score(target, values, average='macro', zero_division=0))
    values = values.reshape(-1)
    if len(values) != len(target) or not np.isfinite(values).all():
        raise ValueError('Prediction length/finite-value contract violated')
    if task == 'ts_forecasting':
        return float(np.mean(np.abs(target - values))) / scale
    return float(np.sqrt(mean_squared_error(target, values))) / scale
