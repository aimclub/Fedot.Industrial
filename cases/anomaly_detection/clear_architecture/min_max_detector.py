from pickle import TRUE
from pandas import array
from sklearn.metrics import f1_score
from anomaly_detection.clear_architecture.settings_args \
    import SettingsArgs
from anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time

from scipy import spatial
import math
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

"""



input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class MinMaxsDetector:
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
        self._print_logs(f"{get_current_time()} Vector detector: Visualisate = {self.args.visualisate}")
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
                temp_output = self.NormalizeData(np.array(temp_output)).tolist()
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
                            self.output_list[i][j+k] = 1
                            next(myiter, None)

    def NormalizeData(self, data):
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
