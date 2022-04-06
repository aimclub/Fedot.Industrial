import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             precision_score, r2_score, roc_auc_score)


class QualityMetric:
    def __init__(self, default_value: float = 0.0):
        self.default_value = default_value

    def metric(self,
               target: list,
               prediction: list) -> float:
        return 0


class RMSE(QualityMetric):

    def metric(self, target, prediction) -> float:
        return mean_squared_error(y_true=target,
                                  y_pred=prediction, squared=False)


class MSE(QualityMetric):

    def metric(self, target, prediction) -> float:
        return mean_squared_error(y_true=target,
                                  y_pred=prediction, squared=True)


class MSLE(QualityMetric):
    def metric(self, target, prediction) -> float:
        return mean_squared_log_error(y_true=target,
                                      y_pred=prediction)


class MAPE(QualityMetric):
    def metric(self, target, prediction) -> float:
        return mean_absolute_percentage_error(y_true=target,
                                              y_pred=prediction)


class F1(QualityMetric):

    def metric(self, target, prediction) -> float:
        self.default_value = 0
        n_classes = np.unique(target)
        n_classes_pred = np.unique(prediction)
        if n_classes.shape[0] > 2 or n_classes_pred.shape[0] > 2:
            additional_params = {'average': 'weighted'}
        else:
            additional_params = {'average': 'binary'}
        return f1_score(y_true=target, y_pred=prediction,
                        **additional_params)


class MAE(QualityMetric):

    def metric(self, target, prediction) -> float:
        return mean_absolute_error(y_true=target, y_pred=prediction)


class R2(QualityMetric):

    def metric(self, target, prediction) -> float:
        return r2_score(y_true=target, y_pred=prediction)


class ROCAUC(QualityMetric):

    def metric(self, target, prediction) -> float:
        self.default_value = 0.5
        n_classes = np.unique(target)
        if n_classes.shape[0] >= 2:
            additional_params = {'multi_class': 'ovr', 'average': 'macro'}
        else:
            additional_params = {}

        score = round(roc_auc_score(y_score=prediction,
                                    y_true=target,
                                    **additional_params), 3)

        return score


class Precision(QualityMetric):
    def metric(self, target, prediction) -> float:
        n_classes = np.unique(target)
        if n_classes.shape[0] >= 2:
            additional_params = {'average': 'macro'}
        else:
            additional_params = {}

        score = round(precision_score(y_pred=prediction,
                                      y_true=target,
                                      **additional_params), 3)
        return score


class Logloss(QualityMetric):
    def metric(self, target, prediction) -> float:
        return log_loss(y_true=target, y_pred=prediction)


class Accuracy(QualityMetric):
    def metric(self, target, prediction) -> float:
        return accuracy_score(y_true=target, y_pred=prediction)
