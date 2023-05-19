import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

anomaly_len = 10
classes = ['Норма', 'Остановка', 'Попадание в лопасти']


class IndustrialPredictor:
    def __init__(self):
        with IndustrialModels():
            self.generator = Pipeline().load('generator/1_pipeline_saved/1_pipeline_saved.json')
        self.predictor = Pipeline().load('predictor/1_pipeline_saved/1_pipeline_saved.json')

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


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../data/power_cons_anomaly.csv')[cols]
    start_random = 1600
    sub_series = series.iloc[start_random:start_random + anomaly_len, :-1]
    sub_series = np.array(sub_series.values.tolist())

    predictor = IndustrialPredictor()
    predict = predictor.predict(sub_series)


if __name__ == '__main__':
    main()
