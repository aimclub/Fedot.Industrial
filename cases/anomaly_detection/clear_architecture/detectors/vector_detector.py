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


class VectorDetector(AbstractDetector):

    def __init__(self, quantile: float,
                 step: int = 2,
                 filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.inner_step = step

        super().__init__(name='Vector Detector')

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
                for i in range(0, len(point_array) - 1, self.inner_step):
                    vector_1 = self._make_vector(last_point, last_point)
                    vector_2 = self._make_vector(point_array[i], last_point)
                    cosine = self._get_angle_between_vectors(
                        last_point,
                        point_array[i]
                    )
                    # bad, without 1 - is better
                    cosine = 1 - spatial.distance.cosine(last_point, point_array[i])
                    cosine_array.append(cosine ** 2)
                avg = sum(cosine_array) / len(cosine_array)
                var = sum((x - avg) ** 2 for x in cosine_array) / len(cosine_array)
                var = np.mean(cosine_array)
                temp_output.append(var)
            if False:
                score_diff = np.diff(temp_output)
                q_95 = np.quantile(temp_output, self.quantile)
                temp_output = list(map(lambda x: 1 if x > q_95 else 0, score_diff))
            # reshaped_data = preprocessing.normalize([np.array(temp_output)]).flatten()
            # reshaped_data = self.NormalizeData(np.array(temp_output)).tolist()
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
                my_iter = iter(range(0, len(self.output_list[i])))
                for j in my_iter:
                    if self.output_list[i][j] == 1:
                        for k in range(self.win_len):
                            self.output_list[i][j + k] = 1
                            next(my_iter, None)
