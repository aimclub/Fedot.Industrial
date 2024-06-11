import math

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.optuna_tuner import OptunaTuner

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.metrics.metrics_implementation import maximised_r2
from fedot_ind.core.operation.decomposition.matrix_decomposition.column_sampling_decomposition import CURDecomposition

dataset = 'kaggle'
result_dict = {}


def create_features(train_data, test_data):
    initial_features = list(test_data.drop(['id'], axis=1).columns)

    unique_vals = []
    for df in [train_data, test_data]:
        for col in initial_features:
            unique_vals += list(df[col].unique())

    unique_vals = list(set(unique_vals))

    for df in [train_data, test_data]:
        df['fsum'] = df[initial_features].sum(axis=1)
        df['fstd'] = df[initial_features].std(axis=1)
        df['special1'] = df['fsum'].isin(np.arange(72, 76))
        df['fskew'] = df[initial_features].skew(axis=1)
        df['fkurtosis'] = df[initial_features].kurtosis(axis=1)

        for i in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            df['q_{}'.format(int(i * 100))
               ] = df[initial_features].quantile(i, axis=1)

        for v in unique_vals:
            df['cnt_{}'.format(v)] = (df[initial_features] == v).sum(axis=1)
    return train_data, test_data


def cur_sampling(tensor, target, projection_rank=0.7):
    projection_rank = math.ceil(max(tensor.shape) * projection_rank)
    decomposer = CURDecomposition(rank=projection_rank)
    sampled_tensor, sampled_target = decomposer.fit_transform(tensor, target)
    return sampled_tensor, sampled_target


if dataset == 'kaggle':
    metric = 'R2'
    task = 'regression'
    model_list = dict(
        # treg=['treg'],
        # scaling_ridge=['scaling', 'ridge'],
        xgbreg=['xgbreg'],
        lgbmreg=['lgbmreg']
    )
    sub_df = pd.read_csv('./examples/big_dataset/sample_submission.csv')
    train_X = pd.read_csv('./examples/big_dataset/train.csv')
    train_y = train_X['FloodProbability'].values
    train_X = train_X.drop(['FloodProbability'], axis=1)
    test_y = train_y

    sub_X = pd.read_csv('./examples/big_dataset/test.csv')
    train_X, sub_X = create_features(train_data=train_X, test_data=sub_X)
    train_X, sub_X = train_X.iloc[:, 1:].values.astype(
        float), sub_X.iloc[:, 1:].values.astype(float)
    test_X = train_X
    sub_X_input = InputData(idx=np.arange(len(sub_X)),
                            features=sub_X,
                            target=np.arange(len(sub_X)),
                            task=Task(TaskTypesEnum.regression),
                            data_type=DataTypesEnum.image)
    tuning_params = dict(
        tuner=OptunaTuner,
        metric=maximised_r2,
        tuning_timeout=15,
        tuning_early_stop=100,
        tuning_iterations=200)
    # sampling_range = [0.1, 0.2, 0.35, 0.5]
    sampling_range = [0.6, 0.75, 0.9]
else:
    train_X = np.load(
        './examples/big_dataset/train_airlinescodrnaadult_fold0.npy')
    train_y = np.load(
        './examples/big_dataset/trainy_airlinescodrnaadult_fold0.npy')
    test_X = np.load(
        './examples/big_dataset/test_airlinescodrnaadult_fold0.npy')
    test_y = np.load(
        './examples/big_dataset/testy_airlinescodrnaadult_fold0.npy')
    metric = 'f1'
    model_list = dict(
        logit=['logit'],
        scaling_rf=[
            'scaling',
            'rf'],
        xgboost=['xgboost'])
    task = 'classification'
    sampling_range = [0.001, 0.01, 0.15, 0.3]

for model_assumption in model_list.keys():
    node_list = model_list[model_assumption]
    iter_dict = {}
    print(f'model-{model_assumption}')
    for share in sampling_range:
        train_X_sampled, train_y_sampled = cur_sampling(
            tensor=train_X, target=train_y, projection_rank=share)
        data_dict = dict(train_data=(train_X_sampled, train_y_sampled),
                         test_data=(test_X, test_y))
        pipeline = AbstractPipeline(task=task,
                                    task_metric=metric)
        result = pipeline.evaluate_pipeline(node_list, data_dict)

        if dataset == 'kaggle':
            r2 = result['quality_metric']
            if r2 > 0:
                result['fitted_model'] = pipeline.tune_pipeline(
                    model_to_tune=result['fitted_model'], tuning_params=tuning_params)
                sub_df['FloodProbability'] = result['fitted_model'].predict(
                    sub_X_input).predict
                sub_df.to_csv(
                    f'./{model_assumption}_sample_percent_{share}_R2_on_train_{r2}.csv',
                    index=False)
        print(f'sampled_rate-{share}')
        print(f'quality_metric_{metric}')
        print(result['quality_metric'])
        iter_dict.update({f'sampled_rate-{share}': result})
    result_dict.update({model_assumption: iter_dict})
_ = 1
