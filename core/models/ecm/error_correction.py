import warnings
from typing import Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot

from core.operation.utils.LoggerSingleton import Logger

warnings.simplefilter(action='ignore', category=FutureWarning)


class Booster:
    """
    Class to implement error correction model (ECM) for classification prediction improvement.

    :param features_train: X_train
    :param target_train: y_train
    :param base_predict: prediction, derived from main model (Quantile, Spectral, Topological, or Wavelet)
    :param timeout: defines the amount of time to compose and tune main prediction model
    :param threshold: parameter used as round boundary for custom_round() method
    :param n_cycles: number of boosting cycles
    :param reshape_flag: ...
    """
    FIRST_CYCLE = 1

    def __init__(self, features_train: np.ndarray,
                 target_train: np.array,
                 base_predict: np.array,
                 timeout: int,
                 threshold: Union[int, float] = 0,
                 reshape_flag: bool = False,
                 n_cycles: int = 3):
        self.logger = Logger().get_logger()
        self.X_train = features_train
        self.target_train = target_train
        self.base_predict = base_predict
        self.threshold = threshold
        # self.timeout = max(3, round(timeout / 4))
        self.timeout = 1
        self.booster_features = {}
        self.check_table = pd.DataFrame()
        self.reshape_flag = reshape_flag
        self.n_cycles = n_cycles
        if self.reshape_flag:
            self.target_train = self.target_train.reshape(-1)

    def run_boosting(self) -> tuple:
        """
        Method to run the boosting process
        """
        self.logger.info('Started boosting')

        target_diff_1 = self.decompose_target(previous_predict=self.base_predict,
                                              previous_target=self.target_train)
        prediction_1, model_1 = self.api_model(target_diff=target_diff_1)
        target_diff_2 = self.decompose_target(previous_predict=prediction_1,
                                              previous_target=target_diff_1)
        prediction_2, model_2 = self.api_model(target_diff=target_diff_2)

        target_diff_3 = self.decompose_target(previous_predict=prediction_2,
                                              previous_target=target_diff_2)
        prediction_3, model_3 = self.api_model(target_diff=target_diff_3)
        self.logger.info('Started boosting ensemble')
        final_prediction, model_ensemble = self.ensemble()
        try:
            self.get_check_table(final_prediction, prediction_1, prediction_2, prediction_3)
        except Exception as ex:
            self.logger.info(f'Problem with saving table: {ex}')

        self.logger.info('Boosting process is finished')
        model_list = [model_1, model_2, model_3]

        return final_prediction, model_list, model_ensemble

    def get_check_table(self, final_prediction, prediction_1, prediction_2, prediction_3) -> None:
        """
        Method to fill self.check_table dataframe with boosting results.

        :param final_prediction: prediction, derived from ensemble of all boosting stages
        :param prediction_1: prediction, derived from first boosting stage
        :param prediction_2: prediction, derived from second boosting stage
        :param prediction_3: prediction, derived from third boosting stage
        :return: None
        """
        try:
            self.check_table['target'] = self.proba_to_vector(self.target_train)
            self.check_table['final_predict'] = self.proba_to_vector(final_prediction)
            self.check_table['1_stage_predict'] = self.proba_to_vector(prediction_1)
            self.check_table['2_stage_predict'] = self.proba_to_vector(prediction_2)
            self.check_table['3_stage_predict'] = self.proba_to_vector(prediction_3)
            self.check_table['base_pred'] = self.proba_to_vector(self.base_predict)
        except Exception as ex:
            self.logger.info(f'Problem with filling table: {ex}')

            self.check_table['base_pred'] = self.base_predict
            self.check_table['final_predict'] = final_prediction
            self.check_table['target'] = self.target_train
            self.check_table['1_stage_predict'] = prediction_1
            self.check_table['2_stage_predict'] = prediction_2
            self.check_table['3_stage_predict'] = prediction_3

    def api_model(self, target_diff: np.ndarray) -> tuple:
        """
        Method used to initiate FEDOT AutoML model to solve regression problem for boosting stage
        """
        self.logger.info(f'Starting cycle {self.FIRST_CYCLE} of boosting')

        fedot_model = Fedot(problem='regression',
                            timeout=self.timeout,
                            seed=20,
                            verbose_level=1,
                            n_jobs=2)

        fedot_model.fit(self.X_train, target_diff)
        prediction = fedot_model.predict(self.X_train)

        if self.reshape_flag:
            prediction = prediction.reshape(-1)

        self.booster_features[self.FIRST_CYCLE] = prediction
        self.FIRST_CYCLE += 1

        return prediction, fedot_model

    @staticmethod
    def decompose_target(previous_predict: np.ndarray,
                         previous_target: np.ndarray):
        """
        Method that returns difference between two arrays: last target and last predict.

        :param previous_predict: last prediction
        :param previous_target: last target
        :return: difference between last target and last predict
        """
        return previous_target - previous_predict

    def ensemble(self) -> tuple:
        """
        Method that ensembles results of all stages of boosting. Depending on number of classes ensemble method
        could be a genetic AutoML model by FEDOT (for binary problem) or SUM method (for multi-class problem)
        """
        self.logger.info('Starting to ensemble boosting results')

        ensemble_model = Fedot(problem='regression',
                               timeout=self.timeout,
                               seed=20,
                               verbose_level=1,
                               n_jobs=6)
        if self.reshape_flag:
            features = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
            target = self.target_train

            ensemble_model.fit(features, target)
            ensemble_prediction = ensemble_model.predict(features)
            ensemble_prediction = ensemble_prediction.reshape(-1)
            return ensemble_prediction, ensemble_model

        else:
            dictlist = []
            for key, value in self.booster_features.items():
                dictlist.append(value)
            ensemble_prediction = sum(dictlist)
            return ensemble_prediction, None

    @staticmethod
    def proba_to_vector(matrix: np.array):
        """
        Method to convert probability matrix to vector of labels
        :type matrix: probability matrix received as a result of prediction_proba for multi-class problem
        """
        vector = np.array([x.argmax() + x[x.argmax()] for x in matrix])
        return vector

    def custom_round(self, num: float) -> int:
        """
        Custom round method with predefined threshold
        :param num: number to be rounded"""
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)


if __name__ == '__main__':
    # Example of usage
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    base_predict = pd.read_csv('base_predict.csv')

    y_train = y_train.values
    base_predict = base_predict.values

    booster = Booster(features_train=X_train,
                      target_train=y_train,
                      base_predict=base_predict,
                      timeout=1)

    result = booster.run_boosting()
