from collections import defaultdict

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.core.operation.transformation.splitter import TSSplitter
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


def get_anomaly_unique(labels, min_anomaly_len=5):
    anomalies = defaultdict(list)
    for i in range(labels.shape[0] - min_anomaly_len):
        subseq = labels[i:i + min_anomaly_len]
        if np.all(subseq == subseq[0]) and subseq[0] != 'Норма':
            anomalies[subseq[0]].append([i, i + min_anomaly_len])

    return anomalies


def split_time_series(series, features_columns: list, target_column: str, is_multivariate=False):
    anomaly_unique = get_anomaly_unique(series[target_column].values, min_anomaly_len=10)

    splitter_unique = TSSplitter(time_series=series[features_columns].values,
                                 anomaly_dict=anomaly_unique,
                                 strategy='unique',
                                 is_multivariate=is_multivariate)

    cls, train_data, test_data = splitter_unique.split(binarize=False)
    return cls, train_data, test_data


def convert_multivar_to_input_data(data):
    concated_df = pd.DataFrame()
    concated_target = []
    for df, target in data:
        concated_df = pd.concat([concated_df, df], axis=0)
        concated_target = concated_target + target
    input_data = InputData(idx=np.arange(len(concated_target)),
                           features=np.array(concated_df.values.tolist()),
                           target=np.array(concated_target).reshape(-1, 1),
                           data_type=DataTypesEnum.image,
                           task=Task(TaskTypesEnum.classification)
                           )
    return input_data


def convert_univar_to_input_data(data):
    concated_df = pd.DataFrame()
    concated_target = []
    for df, target in data:
        concated_df = pd.concat([concated_df, df], axis=0)
        concated_target = concated_target + target
    input_data = InputData(idx=np.arange(len(concated_target)),
                           features=concated_df.values,
                           target=np.array(concated_target).reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    return input_data


def mark_series(series: pd.DataFrame, features_columns: list, target_column: str):
    method = convert_univar_to_input_data
    is_multivariate = False
    if len(features_columns) > 1:
        method = convert_multivar_to_input_data
        is_multivariate = True

    classes, train_data, test_data = split_time_series(series, features_columns, target_column, is_multivariate)

    train_data = method(train_data)
    test_data = method(test_data)
    return train_data, test_data


def main():
    with IndustrialModels():
        cols = ['Power', 'Sound', 'Class']
        series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
        train_data, test_data = mark_series(series, ['Power', 'Sound'], 'Class')
        metrics = {}
        pipeline = PipelineBuilder().add_node('data_driven_basis', branch_idx=0) \
            .add_node('quantile_extractor', branch_idx=0) \
            .add_node('fourier_basis'
                      , branch_idx=1) \
            .add_node('quantile_extractor', branch_idx=1) \
            .add_node('wavelet_basis', branch_idx=2) \
            .add_node('quantile_extractor', branch_idx=2) \
            .join_branches('logit').build()
        pipeline.show()
        # tune pipeline
        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(ClassificationMetricsEnum.f1) \
            .with_iterations(100) \
            .build(train_data)
        pipeline = pipeline_tuner.tune(pipeline)
        pipeline.fit(train_data)

        # calculate metric of tuned pipeline
        predict = pipeline.predict(test_data, output_mode='labels')
        metrics['after_tuned_initial_assumption'] = F1.metric(test_data, predict)
        pipeline.print_structure()

        # magic where we are replacing rf node on node that simply returns merged features
        rf_node = pipeline.nodes[0]
        pipeline.update_node(rf_node, PipelineNode('cat_features'))
        rf_node.nodes_from = []
        rf_node.unfit()
        pipeline.fit(train_data)
        pipeline.save('generator')
        # generate table feature data from train and test samples using "magic" pipeline
        train_data_preprocessed = pipeline.root_node.predict(train_data)
        train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)
        test_data_preprocessed = pipeline.root_node.predict(test_data)
        test_data_preprocessed.predict = np.squeeze(test_data_preprocessed.predict)

        train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                            features=train_data_preprocessed.predict,
                                            target=train_data_preprocessed.target,
                                            data_type=train_data_preprocessed.data_type,
                                            task=train_data_preprocessed.task)
        test_data_preprocessed = InputData(idx=test_data_preprocessed.idx,
                                           features=test_data_preprocessed.predict,
                                           target=test_data_preprocessed.target,
                                           data_type=test_data_preprocessed.data_type,
                                           task=test_data_preprocessed.task)

    model_fedot = Fedot(problem='classification', timeout=5, n_jobs=4, metric=['f1'])
    pipeline = model_fedot.fit(train_data_preprocessed)
    pipeline.save('predictor')
    predict = pipeline.predict(test_data_preprocessed, output_mode='labels')
    metrics['after_fedot_composing'] = F1.metric(test_data, predict)
    pipeline.show()
    print(metrics)


if __name__ == '__main__':
    main()
