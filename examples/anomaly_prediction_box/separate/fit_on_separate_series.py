import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from examples.anomaly_prediction_box.both.fit_on_both_series import mark_series
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
    for col in ['Power', 'Sound']:
        train_data, test_data = mark_series(series, [col], 'Class')
        metrics = {}
        with IndustrialModels():
            pipeline = PipelineBuilder().add_node('data_driven_basis', branch_idx=0) \
                .add_node('quantile_extractor', branch_idx=0) \
                .add_node('fourier_basis'
                          , branch_idx=1) \
                .add_node('quantile_extractor', branch_idx=1) \
                .add_node('wavelet_basis', branch_idx=2) \
                .add_node('quantile_extractor', branch_idx=2) \
                .join_branches('logit').build()
            # tune pipeline
            pipeline_tuner = TunerBuilder(train_data.task) \
                .with_tuner(SimultaneousTuner) \
                .with_metric(ClassificationMetricsEnum.f1) \
                .with_iterations(2) \
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
            pipeline.save(f'generator_for_{col}')
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
        pipeline.save(f'predictor_for_{col}')
        predict = pipeline.predict(test_data_preprocessed, output_mode='labels')
        metrics['after_fedot_composing'] = F1.metric(test_data, predict)
        pipeline.show()
        print(metrics)


if __name__ == '__main__':
    main()