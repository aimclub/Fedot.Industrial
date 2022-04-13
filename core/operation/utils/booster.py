from fedot.api.main import Fedot
from cases.analyzer import PerfomanceAnalyzer

from cases.run.utils import read_tsv, get_logger
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd


def get_score(target, prediction):
    if len(np.unique(target)) > 2:
        return roc_auc_score(target, prediction, multi_class='ovo')
    return roc_auc_score(target, prediction)


class Booster:

    cycles = 0

    def __init__(self,
                 new_target,
                 previous_predict,
                 input_data: tuple = None,  # TRAIN (X, y) subset of whole dataset
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 # fedot_parameters: dict = None,
                 # file_name: str = 'EthanolLevel',
                 ):
        self.input_data = input_data
        self.previous_predict = previous_predict
        self.new_target = new_target
        self.analyzer = PerfomanceAnalyzer()
        self.metrics_name = metrics_name

        # self.fedot_parameters = fedot_parameters
        # self.file_name = file_name
        # self.logger.info(f'Boosting of obtained model for <{self.file_name}> has started')
        self.boost_predictions = {'0': self.previous_predict}

        # self.features = 0
        # self.target = 0

        self.logger = get_logger()

    def run_boosting(self):
        # after every boost
        self.cycles += 1


    def get_boost_model(self):
        self.cycles += 1

        self.logger.info(f'Start cycle {self.cycles} of boosting')

        fedot_model = Fedot(problem='classification',
                            timeout=1,
                            seed=20,
                            verbose_level=2,
                            n_jobs=-1)
        self.logger.info('Reading dataset')
        # data = pd.read_csv(r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\data\EthanolLevel\EthanolLevel_TRAIN.tsv',
        #                        sep='\t',
        #                        header=None)

        # self.features = self.input_data[0]
        # self.target = self.input_data[1]

        X, y = self.input_data[0], self.input_data[1]

        # Split Input data into train and test
        X_train, _, y_train, _ = train_test_split(X,
                                                  y,
                                                  test_size=0.2,
                                                  random_state=np.random.randint(100))
        pipeline = fedot_model.fit(X_train, y_train)

        # if len(np.unique(y_train)) > 2:
        #     prediction = fedot_model.predict_proba(X_test)
        # else:

        prediction = fedot_model.predict(X)
        prediction_proba = fedot_model.predict_proba(X)

        corr = prediction.shape[0]
        diff = y - prediction.reshape(corr, )

        self.boost_predictions[self.cycles] = prediction

        sum_of_predictions = sum(self.boost_predictions.values())

        try:
            metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                      target=y,
                                                      predicted_labels=sum_of_predictions,
                                                      predicted_probs=prediction_proba
                                                      )
        except Exception as ex:
            metrics = 'empty'

        print(f'Metrics with {self.cycles} of Boosting: {metrics}')

#
# if __name__ == '__main__':
#     # data = read_tsv('EthanolLevel')
#
#     Booster().get_boost_model()
