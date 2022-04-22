import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from cases.run.utils import get_logger
from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.pipelines.pipeline import Pipeline


class StaticBooster:
    CYCLES = 1

    def __init__(self,
                 X_train,
                 # X_test,
                 y_train,
                 # y_test,
                 base_predict,
                 timeout,
                 threshold=0):

        self.X_train = X_train
        self.y_train = y_train
        self.base_predict = base_predict
        self.threshold = threshold
        self.timeout = round(timeout / 4)
        self.booster_features = {}
        self.check_table = pd.DataFrame()

        self.logger = get_logger()

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

        self.check_table['target'] = self.y_train
        self.check_table['final_predict'] = final_prediction
        self.check_table['base_pred'] = self.base_predict
        print('3 hundred bucks')
        model_list = [model_1, model_2, model_3]
        final_prediction_round = self.check_table['final_predict'].apply(func=self.custom_round).values.reshape(-1)

        return final_prediction, model_list, model_ensemble

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

        fedot_model = Pipeline(current_node)

        fedot_model.fit(input_data=input_data)
        prediction = fedot_model.predict(input_data=input_data)

        self.booster_features[self.CYCLES] = prediction.predict
        self.CYCLES += 1

        return prediction.predict.reshape(-1), fedot_model

    def decompose_target(self, previous_predict, previous_target):
        return previous_target - previous_predict

    def ensemble(self):
        self.logger.info('Starting to ensembling boosting results')

        x_data = pd.DataFrame.from_dict(self.booster_features, orient='index').T.values
        y_data = self.y_train

        task = Task(TaskTypesEnum.regression)

        input_data = InputData(idx=np.arange(0, len(x_data)),
                               features=x_data,
                               target=y_data, task=task,
                               data_type=DataTypesEnum.table)

        xgboost = PrimaryNode('xgboost')
        fedot_model = Pipeline(xgboost)

        fedot_model.fit(input_data)
        ensemble_prediction = fedot_model.predict(input_data)

        return ensemble_prediction.predict.reshape(-1), fedot_model

    def custom_round(self, num):
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)

# if __name__ == '__main__':
#     dataset = 'Earthquakes'
#     X_test_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_test_feats.csv'
#     X_train_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_train_feats.csv'
#     y_train_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_y_train.csv'
#     y_test_path = r'C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock\Earthquakes_y_test.csv'
#     base_predict_path = r"C:\Users\User\Desktop\work-folder\industrial_ts\IndustrialTS\cock" \
#                         r"\probs_preds_target_Earthquakes.csv "
#
#     booster = Booster(X_train=pd.read_csv(X_train_path, index_col=0),
#                       X_test=pd.read_csv(X_test_path, index_col=0),
#                       y_train=pd.read_csv(y_train_path, index_col=0).values.reshape(-1),
#                       y_test=pd.read_csv(y_test_path, index_col=0).values.reshape(-1),
#                       base_predict=pd.read_csv(base_predict_path, index_col=0)['Preds'].values.reshape(-1),
#                       threshold=0
#                       )
#
#     booster.run_boosting()
