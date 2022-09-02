import numpy as np
from sklearn import preprocessing

from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window
"""


class AreasDetector(AbstractDetector):

    def __init__(self, quantile: float,
                 divider_for_anomaly_len_influence: float,
                 filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.divider = divider_for_anomaly_len_influence

        super().__init__(name='Areas Detector', operation='detection')

    def _do_analysis(self):

        def get_distance(data: list, number: int) -> float:
            if data[0][number] >= data[1][number]:
                return abs(data[0][number] - data[1][number])
            else:
                return abs(data[1][number] - data[0][number])

        def fill_the_gap(length, area, prediction: list) -> list:
            for _ in range(length):
                prediction.append(area)
            return prediction

        self.output_list = []

        for data in self.data:
            odd_new_predicts = []
            state = 0
            counter_for_areas = 0
            for i in range(len(data[0])):
                if state == 0:
                    counter_for_areas = 1
                    area = get_distance(data, i)
                    if data[0][i] - data[1][i] >= 0:
                        state = 1
                    else:
                        state = -1
                    continue
                if state == 1:
                    if data[0][i] - data[1][i] >= 0: 
                        area += get_distance(data, i)
                        counter_for_areas += 1
                    else: 
                        state = -1
                        odd_new_predicts = fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = get_distance(data, i)
                    continue
                if state == -1:
                    if data[0][i] - data[1][i] < 0: 
                        area += get_distance(data, i)
                        counter_for_areas += 1
                    else: 
                        state = 1
                        odd_new_predicts = fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = get_distance(data, i)
                    continue
            odd_new_predicts = fill_the_gap(counter_for_areas, area, odd_new_predicts)
            score_diff = np.diff(odd_new_predicts)
            q_95 = np.quantile(odd_new_predicts, self.quantile)
            max_val = max(odd_new_predicts) + max(odd_new_predicts) / 10
            # odd_new_predicts = list(map(lambda x: max_val if x > q_95 else 0, score_diff))
            reshaped_data = preprocessing.normalize([np.array(odd_new_predicts)]).flatten()
            reshaped_data = self.normalize_data(np.array(odd_new_predicts))
            self.output_list.append(reshaped_data.tolist())
            # self.output_list.append(odd_new_predicts)
