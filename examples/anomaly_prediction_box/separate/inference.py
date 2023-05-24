import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

anomaly_len = 10
classes = ['Норма', 'Остановка', 'Попадание в лопасти']


class UniPredictor:
    def __init__(self, name_of_series: str):
        with IndustrialModels():
            self.generator = Pipeline().load(f'generator_for_{name_of_series}/0_pipeline_saved/0_pipeline_saved.json')
        self.predictor = Pipeline().load(f'predictor_for_{name_of_series}/0_pipeline_saved/0_pipeline_saved.json')

    def predict(self, series: np.ndarray):
        data = InputData(idx=np.array([1]),
                         features=series.reshape(1, -1),
                         target=None,
                         task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)

        with IndustrialModels():
            predict = self.generator.predict(data)
            data_preprocessed = InputData(idx=data.idx,
                                          features=predict.predict,
                                          target=data.target,
                                          data_type=predict.data_type,
                                          task=predict.task)

        predict = self.predictor.predict(data_preprocessed, output_mode='probs').predict

        return predict[0]

class IndustrialPredictor:
    def __init__(self):
        self.cols = ['Power', 'Sound']
        self.uni_predictors = {series_name: UniPredictor(series_name) for series_name in self.cols}

    def predict(self, series: pd.DataFrame):
        prediction = {}
        for col in self.cols:
            prediction[col] = self.uni_predictors[col].predict(series[col].values)
        return prediction


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
    start_random = 1600
    sub_series = series.iloc[start_random:start_random + anomaly_len, :-1]

    predictor = IndustrialPredictor()
    predict = predictor.predict(sub_series)
    print(predict)

if __name__ == '__main__':
    main()
