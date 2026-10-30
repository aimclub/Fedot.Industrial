"""Rehydrate sealed numeric inputs; no raw loaders, downloads or normalization."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..freeze import array_record, verify_freeze


def read_lock(bundle, identity):
    verify_freeze(bundle, check_current=True, expected_seal_sha256=identity)
    lock = json.loads((Path(bundle) / 'lock.json').read_text('utf-8'))
    if lock['schema_version'] != 'industrial_mutation_quality_lock@1' or lock['official_test_used'] is not False:
        raise ValueError('Unsupported frozen quality protocol')
    return lock


def restore_container(saved, role, metadata):
    import torch
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.data.common.enums import StateEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    from fedot.core.repository.dataset_types import DataTypesEnum
    values = {}
    for name in ('features', 'target', 'idx'):
        key = f'{role}_{name}'
        value = np.array(saved[key], copy=True)
        if array_record(value)[1] != metadata['arrays'][key]:
            raise ValueError('Frozen input changed during rehydration')
        values[name] = value if name == 'idx' else torch.as_tensor(value)
    description = metadata['containers'][role]
    task_type = TaskTypesEnum[description['task'].split('.')[-1]]
    horizon = description['horizon']
    task = Task(task_type, TsForecastingParams(horizon)) if horizon is not None else Task(task_type)
    result = TensorData(task=task, data_type=DataTypesEnum[description['data_type'].split('.')[-1]],
                        state=StateEnum[description['state'].split('.')[-1]], ts_forecast_horizon=horizon, **values)
    for name in ('features', 'target', 'idx'):
        if array_record(getattr(result, name))[1] != metadata['arrays'][f'{role}_{name}']:
            raise ValueError('Runtime container altered frozen input')
    return result


def load_frozen_data(bundle, lock, scenario):
    from benchmark.industrial.experiments.evolution_parallelism.data import SearchData
    metadata = lock['datasets'][scenario]
    with np.load(Path(bundle) / f'inputs/{scenario}.npz', allow_pickle=False) as saved:
        folds = tuple((restore_container(saved, f'fold{i}_train', metadata),
                       restore_container(saved, f'fold{i}_validation', metadata))
                      for i in range(lock['protocol']['inner_folds']))
        refit = restore_container(saved, 'refit', metadata)
    # The control container is deliberately unavailable to the search objective.
    return SearchData(folds, refit, None, deepcopy(metadata['source']))


def load_control(bundle, lock, scenario):
    with np.load(Path(bundle) / f'inputs/{scenario}.npz', allow_pickle=False) as saved:
        return restore_container(saved, 'control', lock['datasets'][scenario])
