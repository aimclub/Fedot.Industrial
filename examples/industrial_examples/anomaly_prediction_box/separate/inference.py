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


class UniPredictor:
    def __init__(self,
                 path_to_generator: str = 'generator/1_pipeline_saved/1_pipeline_saved.json',
                 path_to_predictor: str = 'predictor/1_pipeline_saved/1_pipeline_saved.json'):
        """
        Initializes models for generator and predictor

        :param path_to_generator: path to .json file of generator pipeline
        :param path_to_predictor: path to .json file of predictor pipeline
        """
        with IndustrialModels():
            self.generator = Pipeline().load(path_to_generator)
        self.predictor = Pipeline().load(path_to_predictor)

    def fit(self, series: np.ndarray, anomalies: np.ndarray):
        """
        Refit generator and predictor

        :param series: array containing train features (lag windows)
        :param anomalies: array containing anomaly label for each sample in features
        """
        train_data = InputData(idx=np.arrange(series.shape[0]),
                               features=series.reshape(1, -1),
                               target=anomalies,
                               task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
        with IndustrialModels():
            self.generator.unfit()
            self.generator.fit(train_data)
            # generate table feature data from train and test samples using "magic" pipeline
            train_data_preprocessed = self.generator.root_node.predict(train_data)
            train_data_preprocessed.predict = np.squeeze(train_data_preprocessed.predict)
            train_data_preprocessed = InputData(idx=train_data_preprocessed.idx,
                                                features=train_data_preprocessed.predict,
                                                target=train_data_preprocessed.target,
                                                data_type=train_data_preprocessed.data_type,
                                                task=train_data_preprocessed.task)
        self.predictor.unfit()
        self.predictor.fit(train_data_preprocessed)

    def predict(self, series: np.ndarray, output_mode: str = 'labels') -> np.ndarray:
        """
                Refit generator and predictor

                :param series: array containing train features (lag windows)
                :param output_mode: 'labels' - returns only labels, 'default' - returns probabilities

                :returns np.ndarray: predicted anomaly label for each sample
                """
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

        predict = self.predictor.predict(data_preprocessed, output_mode=output_mode).predict

        return predict[0]

    def save(self,
             path_to_generator: str = 'generator',
             path_to_predictor: str = 'predictor'):
        """
        Saves models for generator and predictor

        :param path_to_generator: save path to generator pipeline
        :param path_to_predictor: save path to predictor pipeline
        """
        self.generator.save(path_to_generator)
        self.predictor.save(path_to_predictor)

    def load(self,
             path_to_generator: str = 'generator/1_pipeline_saved/1_pipeline_saved.json',
             path_to_predictor: str = 'predictor/1_pipeline_saved/1_pipeline_saved.json'):
        """
        Loads models for generator and predictor

        :param path_to_generator: path to .json file of generator pipeline
        :param path_to_predictor: path to .json file of predictor pipeline
        """
        with IndustrialModels():
            self.generator = Pipeline().load(path_to_generator)
        self.predictor = Pipeline().load(path_to_predictor)


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