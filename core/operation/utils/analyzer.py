from typing import Dict, List, Union

from sklearn import preprocessing

from core.metrics.metrics_implementation import *
from core.operation.utils.LoggerSingleton import Logger


class PerformanceAnalyzer:
    """Class responsible for calculating metrics for predictions.

    """
    metric_list = ['roc_auc', 'f1', 'precision', 'accuracy', 'logloss']
    logger = Logger().get_logger()

    @staticmethod
    def problem_and_metric_for_dataset(task_type: str) -> Union[List, None]:
        if task_type == 'classification':
            return ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']
        elif task_type == 'regression':
            return ['rmse', 'r2']
        else:
            return None

    @staticmethod
    def check_target(target, predictions):
        if type(target) is not np.ndarray:
            target = np.array(target)
        if type(target[0]) is str:
            enc = preprocessing.LabelEncoder().fit(target)
            converted_target = np.array(enc.transform(target))
            converted_predictions = np.array(enc.transform(predictions))
        else:
            converted_target = target
            converted_predictions = predictions

        if converted_target.dtype == object:
            converted_target = [int(x) for x in converted_target]

        return converted_target, converted_predictions

    def calculate_metrics(self,
                          target: list,
                          metric_list: list = ('roc_auc', 'f1', 'precision', 'accuracy', 'logloss'),
                          predicted_labels: list = None,
                          predicted_probs: list = None) -> Dict:

        metric_dict = dict(roc_auc=ROCAUC, f1=F1, precision=Precision,
                           accuracy=Accuracy, logloss=Logloss, rmse=RMSE,
                           r2=R2, mae=MAE, mse=MSE, mape=MAPE)

        result_metric = []
        for metric_name in metric_list:
            chosen_metric = metric_dict[metric_name]
            try:
                score = chosen_metric(target=target,
                                      predicted_labels=predicted_labels,
                                      predicted_probs=predicted_probs).metric()
                score = round(score, 3)
                result_metric.append(score)
            except Exception as err:
                self.logger.info(f'Score cannot be calculated for {metric_name} metric')
                self.logger.info(err)
                result_metric.append(0)

        result_dict = dict(zip(metric_list, result_metric))

        return result_dict
