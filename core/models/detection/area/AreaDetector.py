import numpy as np
from sklearn import preprocessing

from core.models.detection.AbstractDetector import AbstractDetector


class AreasDetector(AbstractDetector):
    """
    input format:
        dict with "data" and "labels" fields

    Output 
        the same dict but with additional list of window
    """

    def __init__(self, quantile: float,
                 divider_for_anomaly_len_influence: float,
                 filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.divider = divider_for_anomaly_len_influence

        super().__init__(name='Areas Detector', operation='detection')

    def get_distance(self, data: list, number: int) -> float:
        return abs(data[0][number]) + abs(data[1][number])

    def fill_the_gap(self, len, area, prediction: list) -> list:
        for _ in range(len):
            prediction.append(area)
        return prediction

    def _do_analysis(self, method_name='baseline'):

        self.output_list = []

        if method_name == 'baseline':
            self._baseline()
        else:
            self._zero_intersection()
        return self.output_list

    def _baseline(self):

        for data in self.data:
            odd_new_predicts = []
            state = 0
            counter_for_areas = 0
            for i in range(len(data[0])):
                if state == 0:
                    counter_for_areas = 1
                    area = self.get_distance(data, i)
                    if data[0][i] - data[1][i] >= 0:
                        state = 1
                    else:
                        state = -1
                    continue
                if state == 1:
                    if data[0][i] - data[1][i] >= 0:
                        area += self.get_distance(data, i)
                        counter_for_areas += 1
                    else:
                        state = -1
                        odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = self.get_distance(data, i)
                    continue
                if state == -1:
                    if data[0][i] - data[1][i] < 0:
                        area += self.get_distance(data, i)
                        counter_for_areas += 1
                    else:
                        state = 1
                        odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = self.get_distance(data, i)
                    continue
            odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
            reshaped_data = self.normalize_data(np.array(odd_new_predicts))
            self.output_list.append(reshaped_data.tolist())

    def _zero_intersection(self) -> None:

        for data in self.data:
            odd_new_predicts = []
            state = 0
            counter_for_areas = 0
            for i in range(len(data[0])):
                if state == 0:
                    counter_for_areas = 1
                    area = self.get_distance(data, i)
                    if data[0][i] >= 0:
                        state = 1
                    else:
                        state = -1
                    continue
                if state == 1:
                    if data[0][i] >= 0:
                        area += self.get_distance(data, i)
                        counter_for_areas += 1
                    else:
                        state = -1
                        odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = self.get_distance(data, i)
                    continue
                if state == -1:
                    if data[0][i] < 0:
                        area += self.get_distance(data, i)
                        counter_for_areas += 1
                    else:
                        state = 1
                        odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
                        counter_for_areas = 1
                        area = self.get_distance(data, i)
                    continue
            odd_new_predicts = self.fill_the_gap(counter_for_areas, area, odd_new_predicts)
            reshaped_data = preprocessing.normalize([np.array(odd_new_predicts)]).flatten()
            self.output_list.append(reshaped_data.tolist())
