import numpy as np
from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum

from core.repository.initializer_industrial_models import initialize_industrial_models, remove_industrial_models
from tests.unit.repository.test_repo import initialize_uni_data

if __name__ == '__main__':
    np.random.seed(0)
    initialize_industrial_models()
    train_data, test_data = initialize_uni_data()
    industrial = get_operations_for_task(task=train_data.task, mode='data_operation', tags=["extractor", "basis"])
    other = get_operations_for_task(task=train_data.task, forbidden_tags=["basis", "extractor"])
    metrics = {}
    assumptions = [
        PipelineBuilder().add_node('data_driven_basic').add_node('topological_extractor') \
            .add_node('rf').build(),
        PipelineBuilder().add_node('data_driven_basic').add_node('quantile_extractor') \
            .add_node('rf').build(),
        PipelineBuilder().add_node('data_driven_basic').add_node('signal_extractor') \
            .add_node('rf').build(),
        PipelineBuilder().add_node('data_driven_basic').add_node('recurrence_extractor') \
            .add_node('rf').build()
    ]
    pipeline = PipelineBuilder().add_node(
        'data_driven_basic', branch_idx=0).add_node('topological_extractor', branch_idx=0).add_node(
        'data_driven_basic', branch_idx=1).add_node('quantile_extractor', branch_idx=1).add_node(
        'data_driven_basic', branch_idx=2).add_node('signal_extractor', branch_idx=2).add_node(
        'data_driven_basic', branch_idx=3).add_node('recurrence_extractor', branch_idx=3).join_branches('rf').build()

    pipeline_tuner = TunerBuilder(train_data.task) \
        .with_tuner(PipelineTuner) \
        .with_metric(ClassificationMetricsEnum.f1) \
        .with_iterations(2) \
        .build(train_data)
    pipeline = pipeline_tuner.tune(pipeline)
    pipeline.fit(train_data)
    predict = pipeline.predict(test_data, output_mode='labels')
    metrics['before fedot_composing'] = F1.metric(test_data, predict)
    pipeline.print_structure()
    rf_node = pipeline.nodes[0]

    pipeline.update_node(rf_node, SecondaryNode('dummy'))
    rf_node.nodes_from = []
    rf_node.unfit()
    pipeline.fit(train_data)

    train_data_preprocessed = pipeline.root_node.predict(train_data)
    test_data_preprocessed = pipeline.root_node.predict(test_data)

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

    remove_industrial_models()
    fedot_pipeline = Pipeline(rf_node)
    model_fedot = Fedot(problem='classification', timeout=5, n_jobs=4, metric=['f1'], initial_assumption=fedot_pipeline)

    pipeline = model_fedot.fit(train_data_preprocessed)
    predict = pipeline.predict(test_data_preprocessed, output_mode='labels')
    metrics['after_fedot_composing'] = F1.metric(test_data, predict)
    pipeline.show()
    print(metrics)
