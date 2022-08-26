import math

import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window 
"""


class MinMaxDetector(AbstractDetector):

    def __init__(self,
                 quantile: float,
                 step: int = 2,
                 filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.inner_step = step

        super().__init__('Vector Detector')

    def _analysis(self) -> None:
        for dataset in self.windowed_data:
            temp_output = []
            for window in tqdm(dataset, colour="RED"):
                point_array = []
                for i in range(len(window[0])):
                    point = []
                    for j in range(len(window)):
                        point.append(window[j][i])
                    point_array.append(point)

                maximum = max(point_array[0])
                minimum = min(point_array[0])
                for i in range(1, len(point_array), self.inner_step):
                    temp_maximum = max(point_array[i])
                    temp_minimum = min(point_array[i])
                    maximum = max(maximum, temp_maximum)
                    minimum = min(minimum, temp_minimum)
            
                temp_output.append(abs(maximum - minimum))
            if False:
                print(temp_output)
                score_diff = np.diff(temp_output)
                q_95 = np.quantile(temp_output, self.quantile)
                temp_output = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
                print(temp_output)
            else:
                reshaped_data = preprocessing.normalize([np.array(temp_output)]).flatten()
                temp_output = self.normalize_data(np.array(temp_output)).tolist()
            self.output_list.append(temp_output)
        new_output_data = []
        for _ in range(len(self.output_list)):
            new_output_data.append([])
        for i, predict in enumerate(self.output_list):
            temp_predict = []
            goal_len = len(self.data[i][0])
            for j in range(len(predict)):
                for t in range(self.step):
                    temp_predict.append(predict[j])
            for _ in range(len(temp_predict), goal_len):
                temp_predict.append(predict[-1])
            new_output_data[i] = temp_predict
        self.output_list = new_output_data
        if False:
            for i in range(len(self.output_list)):
                my_iterator = iter(range(0, len(self.output_list[i])))
                for j in my_iterator:
                    if self.output_list[i][j] == 1:
                        for k in range(self.win_len):
                            self.output_list[i][j + k] = 1
                            next(my_iterator, None)

    def _get_angle_between_vectors(self, vector1, vector2):
        sum_of_coordinates = 0
        for i in range(len(vector1)):
            sum_of_coordinates += vector1[i] * vector2[i]
        if self._get_vector_len(vector1) * self._get_vector_len(vector2) == 0:
            return 0
        return math.sin(
            sum_of_coordinates /
            (self._get_vector_len(vector1) * self._get_vector_len(vector2)))

    @staticmethod
    def _get_vector_len(vector):
        sum_of_coordinates = 0
        for coordinate in vector:
            sum_of_coordinates += coordinate ** 2
        return math.sqrt(sum_of_coordinates)
