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
                          target: Union[np.ndarray, List],
                          predicted_labels: Union[np.ndarray, list] = None,
                          predicted_probs: np.ndarray = None) -> Dict:

        labels_diff = max(target) - max(predicted_labels)

        if min(predicted_labels) != min(target):
            if min(target) == -1:
                np.place(predicted_labels, predicted_labels == 1, [-1])
                np.place(predicted_labels, predicted_labels == 0, [1])

        if labels_diff > 0:
            predicted_labels = predicted_labels + abs(labels_diff)
        else:
            target = target + abs(labels_diff)

        metric_dict = dict(roc_auc=ROCAUC, f1=F1, precision=Precision,
                           accuracy=Accuracy, logloss=Logloss, rmse=RMSE,
                           r2=R2, mae=MAE, mse=MSE, mape=MAPE)

        result_metric = []
        for metric_name in self.metric_list:
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

        result_dict = dict(zip(self.metric_list, result_metric))

        return result_dict
