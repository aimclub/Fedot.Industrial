import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from fedot.core.log import default_log as Logger
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.pipeline import Pipeline


class StaticBooster:
    CYCLES = 1

    def __init__(self, X_train, y_train, base_predict, timeout, threshold=0):

        self.X_train = X_train
        self.y_train = y_train
        self.base_predict = base_predict
        self.threshold = threshold
        self.timeout = round(timeout / 4)
        self.booster_features = {}
        self.check_table = pd.DataFrame()

        self.logger = Logger(self.__class__.__name__)

    def run_boosting(self):
        self.logger.info('Started boosting')

        target_diff_1 = self.decompose_target(previous_predict=self.base_predict,
                                              previous_target=self.y_train.reshape(-1))
        prediction_1, model_1 = self.api_model(target_diff=target_diff_1, node='linear')

        target_diff_2 = self.decompose_target(previous_predict=prediction_1,
                                              previous_target=target_diff_1)
        prediction_2, model_2 = self.api_model(target_diff=target_diff_2, node='ridge')

        target_diff_3 = self.decompose_target(previous_predict=prediction_2,
                                              previous_target=target_diff_2)
        prediction_3, model_3 = self.api_model(target_diff=target_diff_3, node='lasso')

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

        model_list = [model_1, model_2, model_3]
        final_prediction_round = self.check_table['final_predict'].apply(func=np.round).values.reshape(-1)

        return final_prediction, model_list, model_ensemble

    def proba_to_vector(self, matrix):
        try:
            vector = np.array([x.argmax() + x[x.argmax()] for x in matrix])
            return vector
        except IndexError:
            return matrix

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

    def api_model(self, target_diff, node: str):
        self.logger.info(f'Starting cycle {self.CYCLES} of boosting')

        x_data = np.array(self.X_train)
        y_data = target_diff

        task = Task(TaskTypesEnum.regression)
        input_data = InputData(idx=np.arange(0, len(x_data)),
                               features=x_data,
                               target=y_data,
                               task=task,
                               data_type=DataTypesEnum.table)
        current_node = PrimaryNode(node)

        boosting_model = Pipeline(current_node)

        boosting_model.fit(input_data=input_data)
        prediction = boosting_model.predict(input_data=input_data)

        self.booster_features[self.CYCLES] = prediction.predict.reshape(-1)
        self.CYCLES += 1

        return prediction.predict.reshape(-1), boosting_model

    def decompose_target(self, previous_predict, previous_target):
        return previous_target - previous_predict

    def ensemble(self):
        self.logger.info('Starting to ensembling boosting results')

        x_data = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
        y_data = self.y_train

        task = Task(TaskTypesEnum.regression)

        input_data = InputData(idx=np.arange(0, len(x_data)),
                               features=x_data,
                               target=y_data,
                               task=task,
                               data_type=DataTypesEnum.table)

        xgboost = PrimaryNode('xgboost')
        ensemble_model = Pipeline(xgboost)

        ensemble_model.fit(input_data)
        ensemble_prediction = ensemble_model.predict(input_data)

        if np.unique(y_data).shape[0] > 2:
            ensemble_prediction = np.array([x.argmax() + x[x.argmax()] for x in ensemble_prediction.predict])
        else:
            ensemble_prediction = ensemble_prediction.predict.reshape(-1)

        return ensemble_prediction, ensemble_model

    def custom_round(self, num):
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)
