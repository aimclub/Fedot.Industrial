from typing import Optional, Union

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from golem.core.dag.graph import Graph
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    d2_absolute_error_score,
    explained_variance_score,
    f1_score,
    log_loss,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    # root_mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, median_absolute_scaled_error, \
    mean_absolute_error as tsf_mae, median_absolute_error as tsf_mdae, mean_absolute_percentage_error as tsf_mape

from fedot_ind.core.architecture.settings.computational import backend_methods as np

# from fedot_ind.core.architecture.preprocessing.data_convertor import DataConverter
from fedot_ind.core.metrics.anomaly_detection.function import (
    check_errors,
    single_average_delay,
    single_detecting_boundaries,
    single_evaluate_nab,
)


def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error (MSE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class ParetoMetrics:
    def pareto_metric_list(self,
                           costs: Union[list,
                                        np.ndarray],
                           maximise: bool = True) -> np.ndarray:
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
    def __init__(self,
                 target,
                 predicted_labels,
                 predicted_probs=None,
                 metric_list: list = (
                     'f1', 'roc_auc', 'accuracy', 'logloss', 'precision'),
                 default_value: float = 0.0):
        self.predicted_probs = predicted_probs
        labels_as_matrix = len(predicted_labels.shape) >= 2
        labels_as_one_dim = min(predicted_labels.shape) == 1
        if labels_as_matrix and not labels_as_one_dim:
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
        return root_mean_squared_error(
            y_true=self.target,
            y_pred=self.predicted_labels,
        )


class SMAPE(QualityMetric):
    def metric(self):
        return 1 / len(self.predicted_labels) \
            * np.sum(2 * np.abs(self.target - self.predicted_labels) / (np.abs(self.predicted_labels)
                                                                        + np.abs(self.target)) * 100)


class MSE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_error(
            y_true=self.target,
            y_pred=self.predicted_labels,
        )


class MSLE(QualityMetric):
    def metric(self) -> float:
        return mean_squared_log_error(
            y_true=self.target,
            y_pred=self.predicted_labels)


class MAPE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_percentage_error(
            y_true=self.target, y_pred=self.predicted_labels)


class F1(QualityMetric):
    output_mode = 'labels'

    def metric(self) -> float:
        n_classes = len(np.unique(self.target))
        n_classes_pred = len(np.unique(self.predicted_labels))

        try:
            if n_classes > 2 or n_classes_pred > 2:
                return f1_score(
                    y_true=self.target,
                    y_pred=self.predicted_labels,
                    average='weighted')
            else:
                pos_label = QualityMetric._get_least_frequent_val(self.target)
                return f1_score(
                    y_true=self.target,
                    y_pred=self.predicted_labels,
                    average='binary',
                    pos_label=pos_label)
        except ValueError:
            return self.default_value


class MAE(QualityMetric):
    def metric(self) -> float:
        return mean_absolute_error(
            y_true=self.target,
            y_pred=self.predicted_labels)


class R2(QualityMetric):
    def metric(self) -> float:
        return r2_score(y_true=self.target, y_pred=self.predicted_labels)


def maximised_r2(graph: Graph, reference_data: InputData, **kwargs):
    result = graph.predict(reference_data)
    r2_value = r2_score(y_true=reference_data.target, y_pred=result.predict)
    return 1 - r2_value


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

        score = roc_auc_score(
            y_score=prediction,
            y_true=target,
            labels=np.unique(target),
            **additional_params)
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
            y_pred=self.predicted_labels,
            y_true=self.target,
            **additional_params)
        score = round(score, 3)
        return score


class Logloss(QualityMetric):
    def metric(self) -> float:
        return log_loss(y_true=self.target, y_pred=self.predicted_probs)


class Accuracy(QualityMetric):
    output_mode = 'labels'

    def metric(self) -> float:
        return accuracy_score(y_true=self.target, y_pred=self.predicted_labels)


def _flatten_metric_result(result: dict) -> dict:
    flat = {}
    for key, value in result.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f'{key}_{sub_key}'] = sub_value
        elif isinstance(value, list):
            flat[key] = value
        else:
            flat[key] = value
    return flat


def _result_to_dataframe(result: dict, rounding_order: int) -> pd.DataFrame:
    return pd.DataFrame([_flatten_metric_result(result)], index=[0]).round(rounding_order)


def calculate_regression_metric(target,
                                predicted_labels=None,
                                metric_names=None,
                                rounding_order=3,
                                **kwargs):
    from fedot_ind.core.metrics.metrics import calculate_regression_metric as _calculate

    result = _calculate(
        target=target,
        predicted=predicted_labels,
        metrics=metric_names,
        rounding_order=rounding_order,
        **kwargs,
    )
    if kwargs.get('return_dataframe', True):
        return _result_to_dataframe(result, rounding_order)
    return result


def calculate_forecasting_metric(target,
                                 predicted_labels=None,
                                 metric_names=None,
                                 rounding_order=3,
                                 train_data=None,
                                 seasonality=None,
                                 **kwargs):
    from fedot_ind.core.metrics.metrics import calculate_forecasting_metric as _calculate

    result = _calculate(
        target=target,
        predicted=predicted_labels,
        metrics=metric_names,
        rounding_order=rounding_order,
        train_data=train_data,
        seasonality=seasonality,
        **kwargs,
    )
    if kwargs.get('return_dataframe', True):
        return _result_to_dataframe(result, rounding_order)
    return result


def calculate_classification_metric(target,
                                    predicted_labels=None,
                                    predicted_probs=None,
                                    metric_names=None,
                                    rounding_order=3,
                                    **kwargs):
    from fedot_ind.core.metrics.metrics import calculate_classification_metric as _calculate

    result = _calculate(
        target=target,
        predicted_labels=predicted_labels,
        predicted_probs=predicted_probs,
        metrics=metric_names,
        rounding_order=rounding_order,
        **kwargs,
    )
    if kwargs.get('return_dataframe', True):
        return _result_to_dataframe(result, rounding_order)
    return result


def kl_divergence(solution: pd.DataFrame,
                  submission: pd.DataFrame,
                  epsilon: float = 0.001,
                  micro_average: bool = False,
                  sample_weights: pd.Series = None):
    """
    Calculates the Kullback-Leibler (KL) divergence between the solution and submission dataframes.

    The KL divergence is a measure of how one probability distribution diverges from a second, expected
    probability distribution.

    Args:
        solution (pd.DataFrame): The expected probability distribution.
        submission (pd.DataFrame): The probability distribution to compare.
        epsilon (float, optional): A small value to avoid division by zero or log of zero. Defaults to 0.001.
        micro_average (bool, optional): If True, compute the micro-average (i.e., the total sum for all classes) of
                                        the KL divergence. If False, compute the macro-average (i.e., the unweighted
                                        mean for all classes). Defaults to False.
        sample_weights (pd.Series, optional): An array of weights that are assigned to individual samples.
                                              If None, then samples are equally weighted. Defaults to None.

    Returns:
        float: The average KL divergence between the solution and submission dataframes.

    """

    for col in solution.columns:
        if not pd.api.types.is_float_dtype(solution[col]):
            solution[col] = solution[col].astype(float)

        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices,
                     col] = solution.loc[y_nonzero_indices,
                                         col] * np.log(solution.loc[y_nonzero_indices,
                                                                    col] / submission.loc[y_nonzero_indices,
                                                                                          col])
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())


class DetectionQualityMetric:
    """Бзовый класс для pure detection метрик.
    источник правдя для любого запуска: 
    benchmark, CLI, example, AutoML, тесты API shells.
    """
    default_value = 0.0
    need_to_minimize = False # метрику нужно максимизировать (f1, precision, recall, balanced_accuracy)
    output_mode = 'labels'

    @classmethod
    def _to_1d_int_array(cls, values):
        """приводение входных значений к единому формату: 
        одномерный numpy-массив целых чисел."""
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=int).reshape(-1)

    @classmethod
    def metric(cls, target, predict) -> float:
        """каждый класс-метрика обязан реализовать свой metric, 
        иначе код упадёт явно"""
        raise NotImplementedError


class DetectionAccuracy(DetectionQualityMetric):
    """Point-level accuracy."""
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(accuracy_score(target, predict))


class DetectionBalancedAccuracy(DetectionQualityMetric):
    """Point-level balanced accuracy."""
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(balanced_accuracy_score(target, predict))


class DetectionPrecision(DetectionQualityMetric):
    """Point-level anomaly precision."""
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(precision_score(target, predict, zero_division=0))


class DetectionRecall(DetectionQualityMetric):
    """Point-level anomaly recall."""
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(recall_score(target, predict, zero_division=0))


class DetectionF1(DetectionQualityMetric):
    """Binary point-level F1 для label = 1."""
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(f1_score(target, predict, zero_division=0))


class DetectionF1Macro(DetectionQualityMetric):
    """Macro F1 отдельно для класса 0 и для класса 1 
    с последующим усреднением.
    """
    @classmethod
    def metric(cls, target, predict) -> float:
        target = cls._to_1d_int_array(target)
        predict = cls._to_1d_int_array(predict)
        return float(f1_score(target, predict, average='macro', zero_division=0))


# DETECTION_METRICS = {
#     'accuracy': DetectionAccuracy,
#     'balanced_accuracy': DetectionBalancedAccuracy,
#     'precision': DetectionPrecision,
#     'recall': DetectionRecall,
#     'f1': DetectionF1,
#     'f1_macro': DetectionF1Macro,

#     # TODO: реализовать event-level Delay-Aware detection метрики
#     # 'event_precision': DetectionEventPrecision,
#     # 'event_recall': DetectionEventRecall,
#     # 'event_f1': DetectionEventF1,
#     # 'detection_delay': DetectionDelay,
#     # 'fp_per_series': DetectionFalsePositivePerSeries,
#     # 'nab': DetectionNAB,
# }

# SUPPORTED_DETECTION_METRICS = tuple(DETECTION_METRICS)


# def calculate_detection_metric(
#         target,
#         labels,
#         rounding_order=6,
#         metric_names=None,
#         **kwargs):
#     """Единая функция-фасад для расчёта detection-метрик."""
#     if metric_names is None:
#         metric_names = ('f1_macro', 'balanced_accuracy')

#     unsupported_metrics = set(metric_names) - set(DETECTION_METRICS)
#     if unsupported_metrics:
#         raise ValueError(f'Unsupported detection metrics: {sorted(unsupported_metrics)}')

#     return {
#         name: round(float(DETECTION_METRICS[name].metric(target, labels)), rounding_order)
#         for name in metric_names
#     }


# DETECTION_METRICS_TO_MINIMIZE = tuple(
#     name for name, metric in DETECTION_METRICS.items()
#     if metric.need_to_minimize
# )

# DETECTION_METRICS_TO_MAXIMIZE = tuple(
#     name for name, metric in DETECTION_METRICS.items()
#     if not metric.need_to_minimize
# )


class AnomalyMetric(QualityMetric):
    """Legacy NAB/window-based anomaly metric"""
    def __init__(self,
                 target,
                 predicted_labels,
                 params: Optional[OperationParameters] = None):
        """
        Parameters
        ----------
        self.target: variants:
            or: if one dataset : pd.Series with binary int labels (1 is
            anomaly, 0 is not anomaly);

            or: if one dataset : list of pd.Timestamp of self.target labels, or []
            if haven't labels ;

            or: if one dataset : list of list of t1,t2: left and right
            detection, boundaries of pd.Timestamp or [[]] if haven't labels

            or: if many datasets: list (len of number of datasets) of pd.Series
            with binary int labels;

            or: if many datasets: list of list of pd.Timestamp of self.target labels, or
            self.target = [ts,[]] if haven't labels for specific dataset;

            or: if many datasets: list of list of list of t1,t2: left and right
            detection boundaries of pd.Timestamp;
            If we haven't self.target labels for specific dataset then we must insert
            empty list of labels: self.target = [[[]],[[t1,t2],[t1,t2]]].

            __True labels of anomalies or changepoints.
            It is important to have appropriate labels (CP or
            anomaly) for corresponding self.metric (See later "self.metric")

        self.predicted_labels: variants:
            or: if one dataset : pd.Series with binary int labels
            (1 is anomaly, 0 is not anomaly);

            or: if many datasets: list (len of number of datasets)
            of pd.Series with binary int labels.

            __Predicted labels of anomalies or changepoints.
            It is important to have appropriate labels (CP or
            anomaly) for corresponding self.metric (See later "self.metric")

        self.metric: {'nab', 'binary', 'average_time', 'confusion_matrix'}.
            Default='nab'
            Affects to output (see later: Returns)
            Changepoint problem: {'nab', 'average_time'}.
            Standard AD problem: {'binary', 'confusion_matrix'}.
            'nab' is Numenta Anomaly Benchmark self.metric

            'average_time' is both average delay or time to failure
            depend on situation.

            'binary': FAR, MAR, F1.

            'confusion_matrix' standard confusion_matrix for any point.

        window_width: 'str' for pd.Timedelta
            Width of detection window. Default=None.

        share : float, default=0.1
            The share is needed if window_width = None.
            The width of the detection window in this case is equal
            to a share of the width of the length of self.predicted_labels divided
            by the number of real CPs in this dataset. Default=0.1.

        anomaly_window_destination: {'lefter', 'righter', 'center'}. Default='right'
            The parameter of the location of the detection window relative to the anomaly.
            'lefter'  : the detection window will be on the left side of the anomaly
            'righter' : the detection window will be on the right side of the anomaly
            'center'  : the scoring window will be positioned relative to the center of anom.

        self.clear_anomalies_mode : boolean, default=True.
            True : then the `left value of a Scoring function is Atp and the
            `right is Afp. Only the `first value inside the detection window is taken.
            False: then the `right value of a Scoring function is Atp and the
            `left is Afp. Only the `last value inside the detection window is taken.

        intersection_mode: {'cut left window', 'cut right window', 'both'}.
            Default='cut right window'
            The parameter will be used if the detection windows overlap for
            self.target changepoints, which is generally undesirable and requires a
            different approach than simply cropping the scoring window using
            this parameter.
            'cut left window' : will cut the overlapping part of the left window
            'cut right window': will cut the intersecting part of the right window
            'both'            : will crop the intersecting self.share of both the left
            and right windows

        verbose:  boolean, default=True.
            If True, then output useful information

        plot_figure : boolean, default=False.
            If True, then drawing the score fuctions, detection windows and self.predicted_labelss
            It is used for example, for calibration the self.scale_val.

         self.table_val (self.metric='nab'): pd.DataFrame of specific form. See bellow.
            Application profiles of NAB self.metric.If Default is None:
             self.table_val = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                          [1.0,-0.22,1.0,-1.0],
                                          [1.0,-0.11,1.0,-2.0]])
             self.table_val.index = ['Standard','LowFP','LowFN']
             self.table_val.index.name = "Metric"
             self.table_val.columns = ['A_tp','A_fp','A_tn','A_fn']

        scale (metric='nab'): "default" of "improved". Default="improved".
            Scoring function in NAB self.metric.
            'default'  : standard NAB scoring function
            'improved' : Our function for resolving disadvantages
            of standard NAB scoring function

        scale_val : float > 0. Default=1.0.
            Smoothing factor. The smaller it is,
            the smoother the scoring function is.

        Returns
        ----------
        metrics : value of metrics, depend on metric
            'nab': tuple
                - Standard profile, float
                - Low FP profile, float
                - Low FN profile
            'average_time': tuple
                - Average time (average delay, or time to failure)
                - Missing changepoints, int
                - FPs, int
                - Number of self.target changepoints, int
            'binary': tuple
                - F1 self.metric, float
                - False alarm rate, %, float
                - Missing Alarm Rate, %, float
            'binary': tuple
                - TPs, int
                - TNs, int
                - FPs, int
                - FNS, int

        """
        super().__init__(target=target, predicted_labels=predicted_labels)
        self.metric = params.get('self.metric', 'nab')
        self.anomaly_window_destination = params.get(
            'anomaly_window_destination', 'lefter')
        self.intersection_mode = params.get(
            'intersection_mode', 'cut right window')
        self.scale = params.get('scale', 'improved')
        self.anomaly_window_destination = params.get(
            'anomaly_window_destination', 'lefter')
        self.intersection_mode = params.get(
            'intersection_mode', 'cut right window')
        self.scale = params.get('scale', 'improved')

        self.share = params.get('self.share', 0.1)
        self.scale_val = params.get('self.scale_val', 1)

        self.table_val = params.get(' self.table_val', None)
        self.window_width = params.get('self.window_width', None)
        self.clear_anomalies_mode = params.get(
            'self.clear_anomalies_mode', True)

        # self.target_converter = DataConverter(data=self.target)
        # self.labels_converter = DataConverter(data=self.predicted_labels)

    def _check_sort(self, my_list, input_variant):
        for dataset in my_list:
            if input_variant == 2:
                assert all(np.sort(dataset) == np.array(dataset))
            elif input_variant == 3:
                assert all(np.sort(np.concatenate(dataset))
                           == np.concatenate(dataset))
            elif input_variant == 1:
                assert all(dataset.index.values ==
                           dataset.sort_index().index.values)

    # def __conditional_check(self):
    #     target_list = self.target_converter.is_list
    #     target_series = self.target_converter.is_pandas_series
    #     label_series = self.labels_converter.is_pandas_series
    #     unequal_target_labels = len(self.target) == len(self.predicted_labels)
    #     if any([target_list, target_series, label_series, unequal_target_labels]):
    #         return True
    #     else:
    #         return False

    def __evaluate_nab(self, detecting_boundaries):
        matrix = np.zeros((3, 3))
        result = {}
        metric_names = ['Standard', 'LowFP', 'LowFN']

        for i in range(len(self.predicted_labels)):
            matrix_ = single_evaluate_nab(
                detecting_boundaries[i],
                self.predicted_labels[i],
                table_of_val=self.table_val,
                clear_anomalies_mode=self.clear_anomalies_mode,
                scale=self.scale,
                scale_val=self.scale_val)
            matrix = matrix + matrix_

        for t, profile_name in enumerate(metric_names):
            val = round(
                100 * (matrix[0, t] - matrix[1, t]) / (matrix[2, t] - matrix[1, t]), 2)
            result.update({profile_name: val})
        return result

    def __evaluate_average_time(self, detecting_boundaries):
        missing, detectHistory, FP, all_target_anom = 0, [], 0, 0
        for i in range(len(self.predicted_labels)):
            missing_, detectHistory_, FP_, all_target_anom_ = \
                single_average_delay(detecting_boundaries[i], self.predicted_labels[i],
                                     anomaly_window_destination=self.anomaly_window_destination,
                                     clear_anomalies_mode=self.clear_anomalies_mode)
            missing, detectHistory, FP, all_target_anom = missing + missing_, \
                detectHistory + detectHistory_, FP + FP_, \
                all_target_anom + all_target_anom_

        result = dict(false_positive=int(FP),
                      missed_anomaly=missing,
                      all_detection_hist=np.mean(detectHistory),
                      all_anomaly_history=all_target_anom)
        return result

    def _detect_boundary(self):
        target_map_dict = {1: (self.target, None),
                           2: (None, self.target),
                           3: (None, None)}
        target_series, target_list_ts = target_map_dict[self.input_variant]
        detecting_boundaries = [
            single_detecting_boundaries(
                target_series=target_series[i],
                target_list_ts=target_list_ts,
                predicted_labels=self.predicted_labels[i],
                window_width=self.window_width,
                share=self.share,
                anomaly_window_destination=self.anomaly_window_destination,
                intersection_mode=self.intersection_mode) for i in range(
                len(
                    self.target))]
        return detecting_boundaries

    def metric(self):
        # step 1. Check target and labels
        # if self.__conditional_check():
        #     raise Exception('Incorrect format for self.predicted_labels')
        _metric_dict = {'nab': self.__evaluate_nab,
                        'average_time': self.__evaluate_average_time}
        # final check
        self.input_variant = check_errors(self.target)

        self._check_sort(self.target, self.input_variant)
        self._check_sort(self.predicted_labels, 1)

        # step 2. Detect boundaries
        detecting_boundaries = self._detect_boundary()

        metric_dict = _metric_dict[self.metric](detecting_boundaries)
        return metric_dict

# Legacy public function
# def calculate_detection_metric(target, labels, **kwargs):
#     metric_dict = AnomalyMetric(target=target, predicted_labels=labels).metric()
#     return metric_dict

def calculate_detection_metric(target,
                               predicted_labels=None,
                               predicted_probs=None,
                               metric_names=None,
                               rounding_order=3,
                               **kwargs):
    from fedot_ind.core.metrics.metrics import calculate_detection_metric as _calculate

    result = _calculate(
        target=target,
        predicted_labels=predicted_labels,
        predicted_probs=predicted_probs,
        metrics=metric_names,
        rounding_order=rounding_order,
        **kwargs,
    )
    if kwargs.get('return_dataframe', True):
        return _result_to_dataframe(result, rounding_order)
    return result