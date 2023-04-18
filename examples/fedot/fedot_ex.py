from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.core.optimizer import IndustrialEvoOptimizer
import numpy as np
from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import TaskTypesEnum, Task
from tests.unit.repository.test_repo import initialize_uni_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

if __name__ == '__main__':
    np.random.seed(0)
    mode = 'tuning'
    dataset_list = ['Adiac', 'ArrowHead', 'Mallat', 'Meat', 'Rock']
    # initialize industrial repository
    for dataset_name in dataset_list:
        with IndustrialModels():
            train_data, test_data = initialize_uni_data(dataset_name)

            task = Task(TaskTypesEnum.classification)
            industrial = get_operations_for_task(task=train_data.task, mode='data_operation', tags=["extractor", "basis"])
            other = get_operations_for_task(task=train_data.task, forbidden_tags=["basis", "extractor"])
            metrics = {}

            pipeline = PipelineBuilder().add_node('data_driven_basis', branch_idx=0) \
                .add_node('quantile_extractor', branch_idx=0) \
                .add_node('fourier_basis'
                          , branch_idx=1) \
                .add_node('quantile_extractor', branch_idx=1) \
                .add_node('wavelet_basis', branch_idx=2) \
                .add_node('quantile_extractor', branch_idx=2) \
                .add_node('data_driven_basis', branch_idx=3) \
                .add_node('topological_extractor', branch_idx=3) \
                .add_node('data_driven_basis', branch_idx=4) \
                .add_node('recurrence_extractor', branch_idx=4) \
                .join_branches('logit').build()

            if mode == 'tuning':
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
                print(metrics)
                pipeline.print_structure()
            else:
                pipeline = PipelineBuilder().add_node('fourier_basis', branch_idx=0) \
                    .add_node('quantile_extractor', branch_idx=0).join_branches('logit').build()

                industrial_fedot = Fedot(problem='classification', timeout=20, n_jobs=4, metric=['f1'],
                                         initial_assumption=pipeline, optimizer=IndustrialEvoOptimizer,
                                         available_operations=industrial + ['rf'])
                pipeline = industrial_fedot.fit(train_data)
                industrial_fedot.history.save('ind_history.json')
                predict = industrial_fedot.predict(test_data)

                metrics['before fedot_composing'] = industrial_fedot.get_metrics()['f1']
                print(metrics)
                pipeline.print_structure()
                pipeline.show()

            # magic where we are replacing rf node on node that simply returns merged features
            rf_node = pipeline.nodes[0]
            pipeline.update_node(rf_node, PipelineNode('cat_features'))
            rf_node.nodes_from = []
            rf_node.unfit()
            pipeline.fit(train_data)

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

        model_fedot = Fedot(problem='classification', timeout=20, n_jobs=4, metric=['f1'])

        pipeline = model_fedot.fit(train_data_preprocessed)
        predict = pipeline.predict(test_data_preprocessed, output_mode='labels')
        predict_automl = model_fedot.predict(test_data_preprocessed)
        metrics['after_fedot_composing'] = F1.metric(test_data, predict)
        metrics['after_fedot_composing_auto'] = model_fedot.get_metrics()['f1']
        pipeline.show()
        print(dataset_name)
        print(metrics)
    _ = 1