import warnings
from typing import Union

import numpy as np
import pandas as pd
from fedot.api.main import Fedot

from core.operation.utils.LoggerSingleton import Logger

warnings.simplefilter(action='ignore', category=FutureWarning)


class Booster:
    """Class to implement genetic booster model for basic prediction improvement"""
    CYCLES = 1

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.array,
                 base_predict: np.array,
                 timeout: int,
                 threshold: Union[int, float] = 0,
                 reshape_flag: bool = False):
        """
        :param X_train: X_train
        :param y_train: y_train
        :param base_predict: prediction, derived from main model (Quantile, Spectral, Topological, or Discrete
        :param timeout: defines the amount of time to compose and tune main prediction model
        :param threshold: parameter used as round boundary for custom_round() method
        :param reshape_flag:
        """
        self.logger = Logger().get_logger()
        self.X_train = X_train
        self.y_train = y_train
        self.base_predict = base_predict
        self.threshold = threshold
        self.timeout = max(3, round(timeout / 4))
        self.booster_features = {}
        self.check_table = pd.DataFrame()
        self.reshape_flag = reshape_flag
        if self.reshape_flag:
            self.y_train = self.y_train.reshape(-1)

    def run_boosting(self) -> tuple:
        """Method to run the boosting process"""
        self.logger.info('Started boosting')

        target_diff_1 = self.decompose_target(previous_predict=self.base_predict,
                                              previous_target=self.y_train)
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
        except Exception:
            self.logger.info('Problem with saving table')

        self.logger.info('Boosting process is finished')
        model_list = [model_1, model_2, model_3]

        return final_prediction, model_list, model_ensemble

    def get_check_table(self, final_prediction, prediction_1, prediction_2, prediction_3) -> None:
        """Method to fill self.check_table dataframe with boosting results"""
        try:
            self.check_table['target'] = self.proba_to_vector(self.y_train)
            self.check_table['final_predict'] = self.proba_to_vector(final_prediction)
            self.check_table['1_stage_predict'] = self.proba_to_vector(prediction_1)
            self.check_table['2_stage_predict'] = self.proba_to_vector(prediction_2)
            self.check_table['3_stage_predict'] = self.proba_to_vector(prediction_3)
            self.check_table['base_pred'] = self.proba_to_vector(self.base_predict)
        except Exception:
            self.check_table['base_pred'] = self.base_predict
            self.check_table['final_predict'] = final_prediction
            self.check_table['target'] = self.y_train
            self.check_table['1_stage_predict'] = prediction_1
            self.check_table['2_stage_predict'] = prediction_2
            self.check_table['3_stage_predict'] = prediction_3

    def proba_to_vector(self, matrix: np.array):
        """
        :type matrix: probability matrix received as a result of prediction_proba for multi-class problem
        """
        vector = np.array([x.argmax() + x[x.argmax()] for x in matrix])
        return vector

    def api_model(self, target_diff: np.ndarray) -> tuple:
        """Method used to initiate FEDOT AutoML model to solve regression problem for boosting stage
        """
        self.logger.info(f'Starting cycle {self.CYCLES} of boosting')

        fedot_model = Fedot(problem='regression',
                            timeout=self.timeout,
                            seed=20,
                            verbose_level=1,
                            n_jobs=6)

        fedot_model.fit(self.X_train, target_diff)
        prediction = fedot_model.predict(self.X_train)

        if self.reshape_flag:
            prediction = prediction.reshape(-1)

        self.booster_features[self.CYCLES] = prediction
        self.CYCLES += 1

        return prediction, fedot_model

    def decompose_target(self,
                         previous_predict: np.array,
                         previous_target: np.array):
        """Method that returns difference between two arrays: last target and last predict"""
        return previous_target - previous_predict

    def ensemble(self) -> tuple:
        """Method that ensemble results of all stages of boosting. Depending on number of classes ensemble method
        could be a genetic AutoML model by FEDOT (for binary problem) or SUM method (for multi-class problem)"""
        self.logger.info('Starting to ensemble boosting results')

        ensemble_model = Fedot(problem='regression',
                               timeout=self.timeout,
                               seed=20,
                               verbose_level=1,
                               n_jobs=6)
        if self.reshape_flag:
            features = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
            target = self.y_train

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

    def custom_round(self, num: float) -> int:
        """Custom round method with predefined threshold"""
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)