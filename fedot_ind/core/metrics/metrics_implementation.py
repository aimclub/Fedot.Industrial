from typing import Union
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, \
    mean_squared_error, d2_absolute_error_score, \
    median_absolute_error, r2_score
from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             log_loss, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             precision_score, r2_score, roc_auc_score)


class ParetoMetrics:
    def __init__(self):
        pass

    def pareto_metric_list(self, costs: Union[list, np.ndarray], maximise: bool = True) -> np.ndarray:
        """ Calculates the pareto front for a list of costs.

        Args:
            costs: list of costs. An (n_points, n_costs) array.
            maximise: flag for maximisation or minimisation.

        Returns:
            A (n_points, ) boolean array, indicating whether each point is Pareto efficient

        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = np.any(
                        costs[is_efficient] >= c, axis=1)  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(
                        costs[is_efficient] <= c, axis=1)  # Remove dominated points
        return is_efficient


class QualityMetric:
    def __init__(self, target,
                 predicted_labels,
                 predicted_probs=None,
                 metric_list: list = (
                     'f1', 'roc_auc', 'accuracy', 'logloss', 'precision'),
                 default_value: float = 0.0):
        self.predicted_probs = predicted_probs
        self.predicted_labels = np.array(predicted_labels).flatten()
        self.target = np.array(target).flatten()
        self.metric_list = metric_list
        self.default_value = default_value

    def metric(self) -> float:
        pass


class RMSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(y_true=self.target, y_pred=self.predicted_labels, squared=False)


class MSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(y_true=self.target, y_pred=self.predicted_labels, squared=True)


class MSLE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_log_error(y_true=self.target, y_pred=self.predicted_labels)


class MAPE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_percentage_error(y_true=self.target, y_pred=self.predicted_labels)


class F1(QualityMetric):
    def metric(self) -> float:
        target = self.target
        prediction = self.predicted_labels
        self.default_value = 0.0
        n_classes = len(np.unique(target))
        n_classes_pred = len(np.unique(prediction))
        try:
            if n_classes > 2 or n_classes_pred > 2:
                return f1_score(y_true=target, y_pred=prediction, average='weighted')
            else:
                return f1_score(y_true=target, y_pred=prediction, average='binary')
        except ValueError:
            return self.default_value


class MAE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_error(y_true=self.target, y_pred=self.predicted_labels)


class R2(QualityMetric):
    def metric(self) -> float:
        return r2_score(y_true=self.target, y_pred=self.predicted_labels)


class ROCAUC(QualityMetric):
    def metric(self) -> float:
        n_classes = len(np.unique(self.target))

        self.default_value = 0.5
        if n_classes > 2:
            target = pd.get_dummies(self.target)
            additional_params = {'multi_class': 'ovr', 'average': 'macro'}
            if self.predicted_probs is None:
                prediction = pd.get_dummies(self.predicted_labels)
            else:
                prediction = self.predicted_probs
        else:
            target = self.target
            additional_params = {}
            prediction = self.predicted_labels

        score = roc_auc_score(y_score=prediction,
                              y_true=target, **additional_params)
        score = round(score, 3)

        return score


class Precision(QualityMetric):
    def metric(self) -> float:
        target = self.target
        prediction = self.predicted_labels

        n_classes = np.unique(target)
        if n_classes.shape[0] >= 2:
            additional_params = {'average': 'macro'}
        else:
            additional_params = {}

        score = precision_score(
            y_pred=prediction, y_true=target, **additional_params)
        score = round(score, 3)
        return score


class Logloss(QualityMetric):
    def metric(self) -> float:
        target = self.target
        prediction = self.predicted_probs
        return log_loss(y_true=target, y_pred=prediction)


class Accuracy(QualityMetric):
    def metric(self) -> float:
        target = self.target
        prediction = self.predicted_labels
        return accuracy_score(y_true=target, y_pred=prediction)


def calculate_regression_metric(test_target, labels):
    test_target = test_target.astype(float)
    metric_dict = {'r2_score:': r2_score(test_target, labels),
                   'mean_squared_error:': mean_squared_error(test_target, labels),
                   'root_mean_squared_error:': np.sqrt(mean_squared_error(test_target, labels)),
                   'mean_absolute_error': mean_absolute_error(test_target, labels),
                   'median_absolute_error': median_absolute_error(test_target, labels),
                   'explained_variance_score': explained_variance_score(test_target, labels),
                   'max_error': max_error(test_target, labels),
                   'd2_absolute_error_score': d2_absolute_error_score(test_target, labels)
                   }
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    return df


def calculate_classification_metric(test_target, labels, probs):

    metric_dict = {'accuracy:': Accuracy(test_target, labels, probs).metric(),
                   'f1': F1(test_target, labels, probs).metric(),
                   'roc_auc:': ROCAUC(test_target, labels, probs).metric()
                   }
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    return df
