import numpy as np
from scipy.spatial.distance import pdist, squareform


class TSTransformer:
    def __init__(self, time_series, min_signal_ratio, max_signal_ratio, rec_metric):
        self.time_series = time_series
        self.recurrence_matrix = None
        self.threshold_baseline = [0.95, 0.7]  # cosine
        self.min_signal_ratio = min_signal_ratio
        self.max_signal_ratio = max_signal_ratio
        self.rec_metric = rec_metric

    def ts_to_recurrence_matrix(self,
                                threshold=None):
        distance_matrix = pdist(metric=self.rec_metric, X=self.time_series.T)
        distance_matrix = np.ones(shape=distance_matrix.shape[0]) - distance_matrix
        distance_matrix = self.binarization(distance_matrix, threshold=threshold)
        self.recurrence_matrix = squareform(distance_matrix)
        return self.recurrence_matrix

    def binarization(self, distance_matrix, threshold):
        best_threshold_flag = False
        signal_ratio_list = []
        reccurence_matrix = None
        if threshold is None:
            for threshold_baseline in self.threshold_baseline:
                threshold = threshold_baseline
                tmp_array = np.copy(distance_matrix)
                tmp_array[tmp_array < threshold_baseline] = 0.0
                tmp_array[tmp_array >= threshold_baseline] = 1.0
                signal_ratio = np.where(tmp_array == 0)[0].shape[0] / tmp_array.shape[0]

                if self.min_signal_ratio < signal_ratio < self.max_signal_ratio:
                    best_ratio = signal_ratio
                    reccurence_matrix = tmp_array
                    best_threshold_flag = True
                    if signal_ratio > best_ratio:
                        reccurence_matrix = tmp_array
                else:
                    signal_ratio_list.append(abs(self.max_signal_ratio - signal_ratio))

                del tmp_array

        if not best_threshold_flag:
            distance_matrix[distance_matrix < self.threshold_baseline[0]] = 0.0
            distance_matrix[distance_matrix >= self.threshold_baseline[0]] = 1.0
            reccurence_matrix = distance_matrix
        return reccurence_matrix

    def get_recurrence_metrics(self):
        if self.recurrence_matrix is None:
            return self.ts_to_recurrence_matrix()