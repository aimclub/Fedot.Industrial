from core.metrics.metrics_implementation import *
from typing import Dict, Union, List
import numpy as np
from sklearn import preprocessing


class PerfomanceAnalyzer:
    def __init__(self):
        return

    def problem_and_metric_for_dataset(self, task_type: str) -> Union[List, None]:
        if task_type == 'classification':
            return ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']
        elif task_type == 'regression':
            return ['rmse', 'r2']
        else:
            return None

    def check_target(self, target, predictions):
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
                          metric_list: list,
                          target: list,
                          predicted_labels: list = None,
                          predicted_probs: list = None) -> Dict:

        metric_dict = {
            'roc_auc': ROCAUC(),
            'f1': F1(),
            'precision': Precision(),
            'accuracy': Accuracy(),
            'logloss': Logloss(),
            'rmse': RMSE(),
            'r2': R2(),
            'mae': MAE(),
            'mse': MSE(),
            'mape': MAPE()}

        label_only_metrics = ['f1',
                              'accuracy',
                              'precision']

        result_metric = []

        for metric_name in metric_list:
            predicted = predicted_probs

            if metric_name in label_only_metrics:
                predicted = predicted_labels

            chosen_metric = metric_dict[metric_name]

            try:
                score = round(chosen_metric.metric(target=target, prediction=predicted), 3)
            except Exception:
                score = round(chosen_metric.metric(target=target, prediction=predicted_labels), 3)

            result_metric.append(score)

        result_dict = dict(zip(metric_list, result_metric))

        return result_dict
