"""Inner search folds and a control holdout, exclusively from official TRAIN."""
import numpy as np

from benchmark.industrial.experiments.evolution_parallelism.data import SearchData, array_digest


def load_training_data(task, protocol):
    import torch
    from fedot import create_data
    from fedot.core.data.tensor_data import TensorData
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from fedot.core.repository.dataset_types import DataTypesEnum
    from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sktime.datasets import load_acsf1, load_tecator, load_airline

    split_seed = protocol['split_seed']
    if task == 'ts_forecasting':
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        # Airline is supplied as one series; exclude the established official TEST first.
        official_train = load_airline().to_numpy(dtype=float)[:-12]
        horizon = protocol['forecast_outer_horizon']
        search, outer = official_train[:-horizon], official_train[-horizon:]

        def pair(train, validation):
            joined = np.concatenate((train, validation))
            complete = create_data(joined, target=joined.copy(), task='ts_forecasting',
                                   ts_forecast_horizon=horizon, without_target=True)
            return temporal_tensor_holdout(complete)

        folds = tuple(pair(search[:end], search[end:end + horizon])
                      for end in range(len(search) - 3 * horizon, len(search), horizon))
        refit, control = pair(search, outer)
        source = dict(dataset='Airline', train_sha256=array_digest(search),
                      outer_sha256=array_digest(outer), train_shape=list(search.shape),
                      folds=[{'train_end': len(search) - i * horizon, 'horizon': horizon} for i in (3, 2, 1)])
    else:
        x, y = (load_acsf1 if task == 'classification' else load_tecator)(split='train', return_type='numpy3d')
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y)
        indices = np.arange(len(y))
        train_idx, outer_idx = train_test_split(indices, test_size=protocol['classification_outer_fraction'],
                                                random_state=split_seed, stratify=y if task == 'classification' else None)
        train_idx, outer_idx = np.sort(train_idx), np.sort(outer_idx)
        if task == 'classification':
            encoder = LabelEncoder().fit(y[train_idx])
            labels = encoder.transform(y)
            splitter = StratifiedKFold(3, shuffle=True, random_state=split_seed)
        else:
            labels = y.astype(float)
            splitter = KFold(3, shuffle=True, random_state=split_seed)
        native_task = Task(TaskTypesEnum[task])

        def container(values, target):
            return TensorData(features=torch.as_tensor(values), target=torch.as_tensor(target),
                              idx=np.arange(len(target)), task=native_task, data_type=DataTypesEnum.multi_ts)

        def pair(left, right):
            mean = x[left].mean(axis=(0, 2), keepdims=True)
            scale = x[left].std(axis=(0, 2), keepdims=True)
            scale = np.where(scale > 0, scale, 1.)
            return (container((x[left] - mean) / scale, labels[left]),
                    container((x[right] - mean) / scale, labels[right]))

        folds, fold_records = [], []
        for left, right in splitter.split(x[train_idx], labels[train_idx]):
            left, right = train_idx[left], train_idx[right]
            folds.append(pair(left, right))
            fold_records.append(dict(train_indices=left.tolist(), validation_indices=right.tolist()))
        refit, control = pair(train_idx, outer_idx)
        source = dict(dataset='ACSF1' if task == 'classification' else 'Tecator',
                      train_sha256=array_digest(x[train_idx], labels[train_idx]),
                      outer_sha256=array_digest(x[outer_idx], labels[outer_idx]), train_shape=list(x[train_idx].shape),
                      search_indices=train_idx.tolist(), outer_indices=outer_idx.tolist(), folds=fold_records)
    source.update(split_seed=split_seed, test_used_for_search=False, official_test_used=False,
                  outer_used_for_search=False, outer_role='control_holdout_inside_official_train')
    # The fourth field carries the TRAIN control holdout, never the official TEST.
    return SearchData(tuple(folds), refit, control, source)
