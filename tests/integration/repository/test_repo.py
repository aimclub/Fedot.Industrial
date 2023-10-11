import numpy as np
from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.core.architecture.preprocessing.DatasetLoader import DataLoader
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.core.tuning.search_space import get_industrial_search_space


def test_fedot_multi_series():
    with IndustrialModels():
        train_data, test_data = initialize_multi_data()
        pipeline = PipelineBuilder().add_node('eigen_basis', params={'window_length': None}).add_node(
            'quantile_extractor').add_node(
            'rf').build()
        pipeline.fit(train_data)
        predict = pipeline.predict(test_data, output_mode='labels')
        print(F1.metric(test_data, predict))


def initialize_uni_data(dataset_name: str = 'Lightning7'):
    train_data, test_data = DataLoader(dataset_name).load_data()
    train_data = InputData(idx=np.arange(len(train_data[0])),
                           features=train_data[0].values,
                           target=train_data[1].reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    test_data = InputData(idx=np.arange(len(test_data[0])),
                          features=test_data[0].values,
                          target=test_data[1].reshape(-1, 1),
                          task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.table)

    return train_data, test_data


def initialize_multi_data(dataset_name: str = 'Epilepsy'):
    train_data, test_data = DataLoader(dataset_name).load_data()
    train_data = InputData(idx=np.arange(len(train_data[0])),
                           features=np.array(train_data[0].values.tolist()),
                           target=train_data[1].reshape(-1, 1),
                           task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.image)
    test_data = InputData(idx=np.arange(len(test_data[0])),
                          features=np.array(test_data[0].values.tolist()),
                          target=test_data[1].reshape(-1, 1),
                          task=Task(TaskTypesEnum.classification),
                          data_type=DataTypesEnum.image)
    return train_data, test_data


def test_fedot_uni_series():
    with IndustrialModels():
        train_data, test_data = initialize_uni_data()

        metrics = {}
        for extractor_name in [
            'topological_extractor',
            'quantile_extractor',
            # 'signal_extractor',
            # 'recurrence_extractor'
                               ]:
            pipeline = PipelineBuilder().add_node('data_driven_basis').add_node(extractor_name).add_node(
                'rf').build()
            model = Fedot(problem='classification', timeout=1, initial_assumption=pipeline, n_jobs=1)
            model.fit(train_data)
            model.predict(test_data)
            model.get_metrics()
            model.current_pipeline.show()
        print(metrics)


def test_tuner_fedot_uni_series():
    with IndustrialModels():
        train_data, test_data = initialize_uni_data()
        cv_folds = 3
        search_space = SearchSpace(get_industrial_search_space(1))
        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(ClassificationMetricsEnum.f1) \
            .with_cv_folds(cv_folds) \
            .with_iterations(2) \
            .with_search_space(search_space).build(train_data)

    pipeline = PipelineBuilder().add_node('data_driven_basis', params={'n_components': 2}).add_node(
        'topological_extractor').add_node(
        'rf').build()

    pipeline = pipeline_tuner.tune(pipeline)

    pipeline.print_structure()
    pipeline.fit(train_data)
    pipeline.predict(test_data)