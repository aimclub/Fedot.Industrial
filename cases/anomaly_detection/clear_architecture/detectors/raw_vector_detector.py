import math

import numpy as np
from tqdm import tqdm

from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector
from cases.anomaly_detection.clear_architecture.utils.get_time import get_current_time

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window
"""


class RawVectorDetector(AbstractDetector):
  
    def __init__(self):
        super().__init__(name="RawVectorDetector")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} {self.name}: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict["data_body"]["elected_data"]

    def output_data(self) -> dict:
        self.input_dict["data_body"]["detection"] = self.output_list
        return self.input_dict

    def _do_analysis(self) -> None:
        for dataset in self.windowed_data:
            temp_output = []
            for window in tqdm(dataset, colour="RED"):
                point_array = []
                for i in range(len(window[0])):
                    point = []
                    for j in range(len(window)):
                        point.append(window[j][i])
                    point_array.append(point)
                cosinus_array = []
                last_point = point_array[-1]
                inner_step = 1
                for i in range(0, len(point_array)-1):
                    vector_1 = self._make_vector(last_point, point_array[i])
                    #vector_2 = self._make_vector(last_point, point_array[i + inner_step-1])
                    cosinus = self._get_angle_between_vectors(
                        last_point, 
                        point_array[i]
                    )
                    cosinus_array.append(cosinus)
                avg = sum(cosinus_array) / len(cosinus_array)
                var = sum((x-avg) ** 2 for x in cosinus_array) / len(cosinus_array)
                
                temp_output.append(var ** 2)
            score_diff = np.diff(temp_output)
            q_95 = np.quantile(temp_output, 0.99)
            temp_output = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
            self.output_list.append(temp_output)

    def _get_angle_between_vectors(self, vector1, vector2):
        sum_of_coordinates = 0
        for i in range(len(vector1)):
            sum_of_coordinates += vector1[i] * vector2[i]
        if self._get_vector_len(vector1) * self._get_vector_len(vector2) == 0:
            return 0
        return math.cos(
            sum_of_coordinates /
            (self._get_vector_len(vector1) * self._get_vector_len(vector2)))

    @staticmethod
    def _get_vector_len(vector):
        sum_of_coordinates = 0
        for coordinate in vector:
            sum_of_coordinates += coordinate ** 2
        return math.sqrt(sum_of_coordinates)
