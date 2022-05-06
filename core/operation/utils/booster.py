import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from cases.run.utils import get_logger
from fedot.api.main import Fedot


class Booster:
    CYCLES = 1

    def __init__(self,
                 X_train,
                 y_train,
                 base_predict,
                 timeout,
                 threshold=0,
                 reshape_flag=False
                 ):

        self.X_train = X_train
        self.y_train = y_train
        self.base_predict = base_predict
        self.threshold = threshold
        self.timeout = max(3, round(timeout / 4))
        self.booster_features = {}
        self.check_table = pd.DataFrame()
        self.reshape_flag = reshape_flag
        self.logger = get_logger()
        if self.reshape_flag:
            self.y_train = self.y_train.reshape(-1)

    def run_boosting(self):
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
            self.check_table['target'] = self.proba_to_vector(self.y_train)
            self.check_table['final_predict'] = self.proba_to_vector(final_prediction)
            self.check_table['1_stage_predict'] = self.proba_to_vector(prediction_1)
            self.check_table['2_stage_predict'] = self.proba_to_vector(prediction_2)
            self.check_table['3_stage_predict'] = self.proba_to_vector(prediction_3)
            self.check_table['base_pred'] = self.proba_to_vector(self.base_predict)
        except Exception:
            self.logger.info('Problem with check table')
        self.logger.info('Boosting process is finished')
        model_list = [model_1, model_2, model_3]

        return final_prediction, model_list, model_ensemble

    def proba_to_vector(self, matrix):
        vector = np.array([x.argmax() + x[x.argmax()] for x in matrix])
        return vector

    def evaluate_results(self, target, prediction):
        if target.shape[0] > 2:
            average = 'weighted'
        else:
            average = 'binary'

        self.logger.info('Evaluation results')
        accuracy = accuracy_score(y_true=target,
                                  y_pred=prediction)
        f1 = f1_score(y_true=target,
                      y_pred=prediction,
                      average=average)

        return accuracy, f1

    def api_model(self, target_diff):
        self.logger.info(f'Starting cycle {self.CYCLES} of boosting')

        fedot_model = Fedot(problem='regression',
                            timeout=self.timeout,
                            seed=20,
                            verbose_level=2,
                            n_jobs=6)

        fedot_model.fit(self.X_train, target_diff)
        prediction = fedot_model.predict(self.X_train)

        if self.reshape_flag:
            prediction = prediction.reshape(-1)

        self.booster_features[self.CYCLES] = prediction
        self.CYCLES += 1

        return prediction, fedot_model

    def decompose_target(self, previous_predict, previous_target):
        return previous_target - previous_predict

    def ensemble(self):
        self.logger.info('Starting ensembling boosting results')

        fedot_model = Fedot(problem='regression',
                            timeout=self.timeout,
                            seed=20,
                            verbose_level=2,
                            n_jobs=6)
        if self.reshape_flag:
            features = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
            target = self.y_train

            fedot_model.fit(features, target)
            ensemble_prediction = fedot_model.predict(features)
            ensemble_prediction = ensemble_prediction.reshape(-1)
            return ensemble_prediction, fedot_model
        else:
            dictlist = []
            for key, value in self.booster_features.items():
                dictlist.append(value)
            ensemble_prediction = sum(dictlist)
            # features = np.hstack(self.booster_features.values())
            return ensemble_prediction, None

    def custom_round(self, num):
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)
