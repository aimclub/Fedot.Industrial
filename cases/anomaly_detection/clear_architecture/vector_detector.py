# from pickle import TRUE
# from pandas import array
# from sklearn import preprocessing
import math

import numpy as np
from anomaly_detection.clear_architecture.settings_args \
    import SettingsArgs
from anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time
from scipy import spatial
from sklearn.metrics import f1_score
from tqdm import tqdm

"""
input format:
    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window 
"""


class VectorDetector:
    args: SettingsArgs

    def __init__(self, quantile: float, step: int = 2, filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.inner_step = step

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.windowed_data: list = []
        self.output_list: list = []
        self._print_logs(f"{get_current_time()} Vector detector: settings was set.")
        self._print_logs(f"{get_current_time()} Vector detector: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Vector detector: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Vector detector: Data read!")
        self.input_dict = dictionary
        self.windowed_data = self.input_dict["data_body"]["windows_list"]
        self.step = self.input_dict["data_body"]["window_step"]
        self.len = self.input_dict["data_body"]["window_len"]
        self.data = self.input_dict["data_body"]["elected_data"]
        self.lables = self.input_dict["data_body"]["raw_lables"]
        self.win_len = self.input_dict["data_body"]["window_len"]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Vector detector: Start transforming...")
        self._vector_analysis()
        self._print_logs(f"{get_current_time()} Vector detector: Transforming finished!")

    def output_data(self) -> dict:
        if "detection" in self.input_dict["data_body"]:
            previous_predict = self.input_dict["data_body"]["detection"]
            """
            for i, predict in enumerate(previous_predict):
                for j in range(len(predict)):
                    if predict[j] == 1:
                        self.output_list[i][j] = 1
            """
            for i in range(len(self.output_list)):
                self.output_list[i] = [self.output_list[i]]
            for i in range(len(self.output_list)):
                for j in range(len(previous_predict[i])):
                    self.output_list[i].append(previous_predict[i][j])
        else:
            for i in range(len(self.output_list)):
                self.output_list[i] = [self.output_list[i]]

        self.input_dict["data_body"]["detection"] = self.output_list
        if self.filtering:
            score = []
            for i in range(len(self.output_list)):
                score.append(f1_score(self.lables[i], self.output_list[i], average='macro'))
            print("-------------------------------------")
            main_score = sum(score) / len(score)
            print("Average predict:")
            print(main_score)
            print("-------------------------------------")
        return self.input_dict

    def _vector_analysis(self) -> None:
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
                first_point = point_array[0]
                for i in range(0, len(point_array) - 1, self.inner_step):
                    vector_1 = self._make_vector(last_point, last_point)
                    vector_2 = self._make_vector(point_array[i], last_point)
                    cosinus = self._get_angle_between_vectors(
                        last_point,
                        point_array[i]
                    )
                    # bad, without 1 - is better
                    cosinus = 1 - spatial.distance.cosine(last_point, point_array[i])
                    cosinus_array.append(cosinus ** 2)
                avg = sum(cosinus_array) / len(cosinus_array)
                var = sum((x - avg) ** 2 for x in cosinus_array) / len(cosinus_array)
                var = np.mean(cosinus_array)
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
                myiter = iter(range(0, len(self.output_list[i])))
                for j in myiter:
                    if self.output_list[i][j] == 1:
                        for k in range(self.win_len):
                            self.output_list[i][j + k] = 1
                            next(myiter, None)

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _make_vector(self, point_1: list, point_2: list):
        if len(point_1) != len(point_2): raise ValueError("Vectors has to be the same len!")
        vector = []
        for i in range(len(point_1)):
            vector.append(point_2[i] - point_1[i])
        return vector

    def _get_angle_between_vectors(self, vector1, vector2):
        sum_of_coordinates = 0
        for i in range(len(vector1)):
            sum_of_coordinates += vector1[i] * vector2[i]
        if self._get_vector_len(vector1) * self._get_vector_len(vector2) == 0:
            return 0
        return math.sin(
            sum_of_coordinates /
            (self._get_vector_len(vector1) * self._get_vector_len(vector2)))

    def _get_vector_len(self, vector):
        sum_of_coordinates = 0
        for coordinate in vector:
            sum_of_coordinates += coordinate ** 2
        return math.sqrt(sum_of_coordinates)

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
