import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

anomaly_len = 10
classes = ['Норма', 'Остановка', 'Попадание в лопасти']


class IndustrialPredictor:
    """
    Anomalies predictor
    """

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
        train_data = InputData(idx=np.arange(series.shape[0]),
                               features=series,
                               target=anomalies.reshape(-1, 1),
                               task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.image)
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
        model_fedot = Fedot(problem='classification', timeout=5, n_jobs=4, metric=['f1'])
        self.predictor = model_fedot.fit(train_data_preprocessed)

        individuals_with_positions \
            = list({ind.graph.descriptive_id: (ind, gen_num, ind_num)
                    for gen_num, gen in enumerate(model_fedot.history.generations)
                    for ind_num, ind in reversed(list(enumerate(gen)))}.values())

        top_individuals = sorted(individuals_with_positions,
                                 key=lambda pos_ind: pos_ind[0].fitness, reverse=True)[:3]
        _ = 1

    def predict(self, series: np.ndarray, output_mode: str = 'labels') -> np.ndarray:
        """
        Refit generator and predictor

        :param series: array containing train features (lag windows)
        :param output_mode: 'labels' - returns only labels, 'default' - returns probabilities

        :returns np.ndarray: predicted anomaly label for each sample
        """
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

        predict = self.predictor.predict(data_preprocessed, output_mode=output_mode).predict

        return predict[0]

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


def main():
    cols = ['Power', 'Sound', 'Class']
    series = pd.read_csv('../../data/power_cons_anomaly.csv')[cols]
    start_random = 1600
    sub_series = series.iloc[start_random:start_random + anomaly_len, :-1]
    sub_series = np.array(sub_series.values.tolist())

    predictor = IndustrialPredictor(path_to_generator='generator/1_pipeline_saved/1_pipeline_saved.json',
                                    path_to_predictor='predictor/1_pipeline_saved/1_pipeline_saved.json')

    predict = predictor.predict(sub_series)
    series = np.load('data.npy')
    labels = np.load('labels.npy')
    predictor.fit(series, labels)
    predict = predictor.predict(sub_series)



if __name__ == '__main__':
    main()

