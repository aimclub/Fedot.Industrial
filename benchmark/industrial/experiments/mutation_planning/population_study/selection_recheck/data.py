"""Load only original TRAIN and enforce the held-out data access boundary."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

from ..data import file_hash


@dataclass(frozen=True)
class TrainingData:
    train: object
    validation: object
    scale: float
    source: dict


@contextmanager
def training_data_guard(config, scenario):
    """Reject TEST datasets and historical mixed result containers before open."""
    active = [True]
    accesses = {'opened_training_files': [], 'blocked': []}
    forbidden = Path(config['forecast_test']).resolve()

    def audit(event, args):
        if not active[0] or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        name = path.name.lower()
        raw = path.suffix.lower() in ('.ts', '.csv', '.tsv', '.arff', '.npy', '.npz')
        test_name = '_test.' in name or '-test.' in name or name.startswith('test.')
        mixed = name in ('records.json', 'result.json')
        if path == forbidden or (raw and test_name) or mixed:
            accesses['blocked'].append(str(path))
            raise RuntimeError('TRAIN-only diagnostic forbids official TEST and mixed historical results')
        training_name = '_train.' in name or '-train.' in name or name.startswith('train.')
        if raw and training_name:
            accesses['opened_training_files'].append(str(path))

    sys.addaudithook(audit)
    try:
        yield accesses
    finally:
        active[0] = False


def load_training_data(config, scenario, seed, *, forecast_horizon=None):
    """Keep the original inner holdout without opening/hashing the official TEST."""
    from sklearn.model_selection import train_test_split
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    import torch

    task, name = scenario['task'], scenario['name']
    if task == 'ts_forecasting':
        from fedot import create_data
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        if type(forecast_horizon) is not int or forecast_horizon < 1:
            raise ValueError('TRAIN-only forecasting requires an explicit frozen positive horizon')
        path = Path(config['forecast_train'])
        with path.open(newline='', encoding='utf-8') as stream:
            row = next((row for row in csv.reader(stream) if row and row[0] == name), None)
        if row is None:
            raise ValueError(f'Missing training series {name}')
        history = np.asarray([float(value) for value in row[1:] if value.strip()], dtype=float)
        if len(history) <= 2 * forecast_horizon or not np.isfinite(history).all():
            raise ValueError('Require finite training history with an inner holdout')
        data = create_data(history, target=history,
                           task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_horizon)),
                           ts_forecast_horizon=forecast_horizon, without_target=True)
        train, validation = temporal_tensor_holdout(data)
        scale = float(np.mean(np.abs(np.diff(history[:-forecast_horizon]))))
        split = {'train_end': len(history) - forecast_horizon, 'validation_end': len(history),
                 'forecast_horizon': forecast_horizon}
    else:
        from ..supervised_diagnostics import _train_arrays
        x, y, path = _train_arrays(config, scenario)
        if task == 'classification':
            y = y.astype(np.int64)
        indices = np.arange(len(y))
        tr, val = train_test_split(indices, test_size=.25, random_state=seed,
                                   stratify=y if task == 'classification' else None)

        def tensor(idx):
            features = x[idx, None, :] if x.ndim == 2 else x[idx]
            return TensorData(features=torch.as_tensor(features, dtype=torch.float32),
                              target=torch.as_tensor(y[idx]), idx=idx, data_type=DataTypesEnum.ts,
                              task=Task(TaskTypesEnum[task]), state='fit')

        train, validation = tensor(tr), tensor(val)
        scale = 1. if task == 'classification' else float(np.std(y[tr]))
        split = {'train_indices': tr.tolist(), 'validation_indices': val.tolist(), 'official_train_count': len(y)}
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Require a positive training-only normalization scale')
    source = {'scope': 'TRAIN-only', 'files': {str(path): file_hash(path)}, 'split': split,
              'normalization_scale': scale, 'dataset': name, 'task': task}
    return TrainingData(train, validation, scale, source)
