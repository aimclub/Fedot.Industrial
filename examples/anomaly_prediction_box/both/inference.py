import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

anomaly_len = 10
classes = ['Норма', 'Остановка', 'Попадание в лопасти']


class IndustrialPredictor:
    def __init__(self):
        with IndustrialModels():
            self.generator = Pipeline().load('generator/1_pipeline_saved/1_pipeline_saved.json')
        self.predictor = Pipeline().load('predictor/1_pipeline_saved/1_pipeline_saved.json')

    def fit(self, series: np.ndarray, anomalies: np.ndarray):
        """Fit generator and predictor"""
        train_data = InputData(idx=np.array([1]),
                               features=np.expand_dims(series, axis=0).swapaxes(1, 2),
                               target=anomalies,
                               task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.image)
        with IndustrialModels():
            self.generator.unfit()
            pipeline_tuner = TunerBuilder(train_data.task) \
                .with_tuner(SimultaneousTuner) \
                .with_metric(ClassificationMetricsEnum.f1) \
                .with_iterations(2) \
                .build(train_data)
            self.generator = pipeline_tuner.tune(self.generator)
            self.generator.fit(train_data)

            rf_node = self.generator.nodes[0]
            self.generator.update_node(rf_node, PipelineNode('cat_features'))
            rf_node.nodes_from = []
            rf_node.unfit()
            self.generator.fit(train_data)
            # generate table feature data from train and test samples using "magic" pipeline
            train_data_preprocessed = self.generator.root_node.predict(train_data)
            train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)
            train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                                features=train_data_preprocessed.predict,
                                                target=train_data_preprocessed.target,
                                                data_type=train_data_preprocessed.data_type,
                                                task=train_data_preprocessed.task)
        model_fedot = Fedot(problem='classification', timeout=5, n_jobs=4, metric=['f1'])
        self.predictor = model_fedot.fit(train_data_preprocessed)

    def predict(self, series: np.ndarray):
        data = InputData(idx=np.array([1]),
                         features=np.expand_dims(series, axis=0).swapaxes(1, 2),
                         target=None,
                         task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.image)

        with IndustrialModels():
            predict = self.generator.predict(data)
            data_preprocessed = InputData(idx=data.idx,
                                          features=predict.predict,
                                          target=data.target,
                                          data_type=predict.data_type,
                                          task=predict.task)

        predict = self.predictor.predict(data_preprocessed, output_mode='labels').predict

        return predict[0]

    def save(self):
        self.generator.save('generator')
        self.predictor.save('predictor')


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
    start_random = 1600
    sub_series = series.iloc[start_random:start_random + anomaly_len, :-1]
    sub_series = np.array(sub_series.values.tolist())

    predictor = IndustrialPredictor()
    predict = predictor.predict(sub_series)


if __name__ == '__main__':
    main()
