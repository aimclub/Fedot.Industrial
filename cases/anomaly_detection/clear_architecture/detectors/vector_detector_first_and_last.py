import numpy as np
from scipy import spatial
from tqdm import tqdm

from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window
"""


class VectorDetectorFaL(AbstractDetector):

    def __init__(self, quantile: float):
        self.quantile = quantile
        super().__init__(name='VectorDetectorFaL')

    def output_data(self) -> dict:
        if "detection" in self.input_dict["data_body"]:
            previous_predict = self.input_dict["data_body"]["detection"]
            for i, predict in enumerate(previous_predict):
                for j in range(len(predict)):
                    if predict[j] == 1:
                        self.output_list[i][j] = 1
        self.input_dict["data_body"]["detection"] = self.output_list
        self._do_score()
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
                cosine_array = []
                last_point = point_array[-1]
                first_point = point_array[0]
                inner_step = 1
                for i in range(0, len(point_array) // 2):
                    first_point = point_array[i]
                    last_point = point_array[len(point_array) - 1 - i]
                    res = spatial.distance.cosine(last_point, first_point)
                    cosine_array.append(res)
                avg = sum(cosine_array) / len(cosine_array)
                var = sum((x - avg) ** 2 for x in cosine_array) / len(cosine_array)
                cosine = self._get_angle_between_vectors(
                    last_point,
                    first_point
                )
                result = spatial.distance.cosine(last_point, first_point)

                temp_output.append(var ** 2)
            score_diff = np.diff(temp_output)
            q_95 = np.quantile(temp_output, self.quantile)
            temp_output = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
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
