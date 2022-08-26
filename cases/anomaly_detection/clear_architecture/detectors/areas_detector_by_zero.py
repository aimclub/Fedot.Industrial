import numpy as np
from sklearn import preprocessing

from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window
"""


class AreasDetectorByZero(AbstractDetector):

    def __init__(self, quantile: float, divider_for_anomaly_len_influence: float, filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.divider = divider_for_anomaly_len_influence

        super().__init__(name="AreasDetectorByZero")

    def _do_analysis(self) -> None:

        def get_distance(data: list, number: int) -> float:
            return abs(data[0][number]) + abs(data[1][number])

        def fill_the_gap(len, area, prediction: list) -> list:
            for _ in range(len):
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
                    if data[0][i] >= 0:
                        state = 1
                    else:
                        state = -1
                    continue
                if state == 1:
                    if data[0][i] >= 0: 
                        area += get_distance(data, i)
                        counter_for_areas += 1
                    else: 
                        state = -1
                        odd_new_predicts = fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = get_distance(data, i)
                    continue
                if state == -1:
                    if data[0][i] < 0: 
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
            max_val = max(odd_new_predicts) + max(odd_new_predicts)/10
            # odd_new_predicts = list(map(lambda x: max_val if x > q_95 else 0, score_diff))
            reshaped_data = preprocessing.normalize([np.array(odd_new_predicts)]).flatten()
            self.output_list.append(reshaped_data.tolist())
            # self.output_list.append(odd_new_predicts)
