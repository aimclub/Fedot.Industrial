from typing import Union

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             log_loss, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             precision_score, r2_score, roc_auc_score)
from sklearn.metrics import d2_absolute_error_score, explained_variance_score, max_error, median_absolute_error

from fedot_ind.core.architecture.settings.computational import backend_methods as np


class ParetoMetrics:
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
                 metric_list: list = ('f1', 'roc_auc', 'accuracy', 'logloss', 'precision'),
                 default_value: float = 0.0):
        self.predicted_probs = predicted_probs
        if len(predicted_labels.shape) >= 2:
            self.predicted_labels = np.argmax(predicted_labels, axis=1)
        else:
            self.predicted_labels = np.array(predicted_labels).flatten()
        self.target = np.array(target).flatten()
        self.metric_list = metric_list
        self.default_value = default_value

    def metric(self) -> float:
        pass

    @staticmethod
    def _get_least_frequent_val(array: np.ndarray):
        """ Returns the least frequent value in a flattened numpy array. """
        unique_vals, count = np.unique(np.ravel(array), return_counts=True)
        least_frequent_idx = np.argmin(count)
        least_frequent_val = unique_vals[least_frequent_idx]
        return least_frequent_val


class RMSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(y_true=self.target, y_pred=self.predicted_labels, squared=False)


class SMAPE(QualityMetric):
    def metric(self):
        return 1 / len(self.predicted_labels) \
            * np.sum(2 * np.abs(self.target - self.predicted_labels) / (np.abs(self.predicted_labels)
                                                                        + np.abs(self.target)) * 100)


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
    output_mode = 'labels'

    def metric(self) -> float:
        n_classes = len(np.unique(self.target))
        n_classes_pred = len(np.unique(self.predicted_labels))

        try:
            if n_classes > 2 or n_classes_pred > 2:
                return f1_score(y_true=self.target, y_pred=self.predicted_labels, average='weighted')
            else:
                pos_label = QualityMetric._get_least_frequent_val(self.target)
                return f1_score(y_true=self.target, y_pred=self.predicted_labels, average='binary', pos_label=pos_label)
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
            prediction = self.predicted_probs

        score = roc_auc_score(y_score=prediction,
                              y_true=target, labels=np.unique(target), **additional_params)
        score = round(score, 3)

        return score


class Precision(QualityMetric):
    output_mode = 'labels'

    def metric(self) -> float:
        n_classes = np.unique(self.target)
        if n_classes.shape[0] >= 2:
            additional_params = {'average': 'macro'}
        else:
            additional_params = {}

        score = precision_score(
            y_pred=self.predicted_labels, y_true=self.target, **additional_params)
        score = round(score, 3)
        return score


class Logloss(QualityMetric):
    def metric(self) -> float:
        return log_loss(y_true=self.target, y_pred=self.predicted_probs)


class Accuracy(QualityMetric):
    output_mode = 'labels'

    def metric(self) -> float:
        return accuracy_score(y_true=self.target, y_pred=self.predicted_labels)


def calculate_regression_metric(target,
                                labels,
                                rounding_order=3,
                                metric_names=('r2', 'rmse', 'mae'),
                                **kwargs):
    target = target.astype(float)

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    metric_dict = {'r2': r2_score,
                   'mse': mean_squared_error,
                   'rmse': rmse,
                   'mae': mean_absolute_error,
                   'msle': mean_squared_log_error,
                   'mape': mean_absolute_percentage_error,
                   'median_absolute_error': median_absolute_error,
                   'explained_variance_score': explained_variance_score,
                   'max_error': max_error,
                   'd2_absolute_error_score': d2_absolute_error_score}

    df = pd.DataFrame({name: func(target, labels) for name, func in metric_dict.items()
                       if name in metric_names},
                      index=[0])
    return df.round(rounding_order)


def calculate_classification_metric(target,
                                    labels,
                                    probs,
                                    rounding_order=3,
                                    metric_names=('f1', 'roc_auc', 'accuracy')):
    metric_dict = {'accuracy': Accuracy,
                   'f1': F1,
                   'roc_auc': ROCAUC,
                   'precision': Precision,
                   'logloss': Logloss}

    df = pd.DataFrame({name: func(target, labels, probs).metric() for name, func in metric_dict.items()
                       if name in metric_names},
                      index=[0])
    return df.round(rounding_order)


def kl_divergence(solution: pd.DataFrame,
                  submission: pd.DataFrame,
                  epsilon: float = 0.001,
                  micro_average: bool = False,
                  sample_weights: pd.Series = None):
    # Overwrite solution for convenience
    for col in solution.columns:
        # Prevent issue with populating int columns with floats
        if not pd.api.types.is_float_dtype(solution[col]):
            solution[col] = solution[col].astype(float)

        # Clip both the min and max following Kaggle conventions for related metrics like log loss
        # Clipping the max avoids cases where the loss would be infinite or undefined, clipping the min
        # prevents users from playing games with the 20th decimal place of predictions.
        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices, col] = solution.loc[y_nonzero_indices, col] * np.log(
            solution.loc[y_nonzero_indices, col] / submission.loc[y_nonzero_indices, col])
        # Set the loss equal to zero where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())
