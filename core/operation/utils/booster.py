import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from cases.analyzer import PerfomanceAnalyzer
from cases.run.utils import get_logger
from fedot.api.main import Fedot

import numpy as np
import timeit


class Booster:
    CYCLES = 1

    # def __init__(self,
    #              metrics_name: list = ('f1',
    #                                    'roc_auc',
    #                                    'accuracy',
    #                                    'logloss',
    #                                    'precision'),
    #              ):
    #     self.metrics_name = metrics_name

    def __init__(self, X_train, X_test, y_train, y_test, base_predict, threshold):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.base_predict = base_predict
        self.threshold = threshold

        self.booster_features = {}
        self.check_table = pd.DataFrame()

        self.logger = get_logger()

    def run_boosting(self):
        self.logger.info('Started boosting')
        accu_before_boost, f1_before_boost = self.evaluate_results(self.y_test, self.base_predict)
        self.logger.info(f'Before boosting: Accuracy={accu_before_boost}, F1={f1_before_boost}')

        target_diff_1 = self.decompose_target(previous_predict=self.base_predict,
                                              previous_target=self.y_test.reshape(-1))
        prediction_1 = self.api_model(target_diff=target_diff_1).reshape(-1)

        target_diff_2 = self.decompose_target(previous_predict=prediction_1,
                                              previous_target=target_diff_1)
        prediction_2 = self.api_model(target_diff=target_diff_2).reshape(-1)

        target_diff_3 = self.decompose_target(previous_predict=prediction_2,
                                              previous_target=target_diff_2)
        prediction_3 = self.api_model(target_diff=target_diff_3).reshape(-1)

        final_prediction = self.ensemble()

        # accu_after_boost, f1_after_boost = self.evaluate_results(self.y_test, final_prediction)
        # self.logger.info(f'Before boosting: Accuracy={accu_after_boost}, F1={f1_after_boost}')

        self.check_table['target'] = self.y_test
        self.check_table['final_predict'] = final_prediction
        self.check_table['base_pred'] = self.base_predict
        print('3 hundred bucks')
        return final_prediction

    def evaluate_results(self, target, prediction):
        self.logger.info('Evaluation results')
        accuracy = accuracy_score(y_true=target,
                                  y_pred=prediction)
        f1 = f1_score(y_true=target,
                      y_pred=prediction)

        return accuracy, f1

    def api_model(self, target_diff):
        self.logger.info(f'Starting cycle {self.CYCLES} of boosting')

        fedot_model = Fedot(problem='regression',
                            timeout=5,
                            seed=20,
                            verbose_level=2,
                            n_jobs=-1)

        fedot_model.fit(self.X_test, target_diff)
        prediction = fedot_model.predict(self.X_test).reshape(-1)

        self.booster_features[self.CYCLES] = prediction
        self.CYCLES += 1

        return prediction

    def decompose_target(self, previous_predict, previous_target):
        return previous_target - previous_predict

    def ensemble(self):
        fedot_model = Fedot(problem='regression',
                            timeout=2,
                            seed=20,
                            verbose_level=2,
                            n_jobs=-1)

        features = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
        target = self.y_test

        fedot_model.fit(features, target)
        ensemble_prediction = fedot_model.predict(features).reshape(-1)

        return ensemble_prediction

    def custom_round(self, num):
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)


if __name__ == '__main__':
    dataset = 'Earthquakes'
    X_test_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_test_feats.csv'
    X_train_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_train_feats.csv'
    y_train_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_y_train.csv'
    y_test_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_y_test.csv'
    base_predict_path = r"C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock" \
                        r"\probs_preds_target_Earthquakes.csv "

    booster = Booster(X_train=pd.read_csv(X_train_path, index_col=0),
                      X_test=pd.read_csv(X_test_path, index_col=0),
                      y_train=pd.read_csv(y_train_path, index_col=0).values.reshape(-1),
                      y_test=pd.read_csv(y_test_path, index_col=0).values.reshape(-1),
                      base_predict=pd.read_csv(base_predict_path, index_col=0)['Preds'].values.reshape(-1)
                      )

    booster.run_boosting()
