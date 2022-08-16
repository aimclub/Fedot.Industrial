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
class AreasDetector:
    args: SettingsArgs

    def __init__(self, quantile: float, divider_for_anomaly_len_influence: float, filtering: bool = True):
        self.quantile = quantile
        self.filtering = filtering
        self.divider = divider_for_anomaly_len_influence

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.windowed_data: list = []
        self.output_list: list = []
        self._print_logs(f"{get_current_time()} Areas detector: settings was set.")
        self._print_logs(f"{get_current_time()} Areas detector: Visualisate = {self.args.visualisate}")
        self._print_logs(f"{get_current_time()} Areas detector: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Areas detector: Data read!")
        self.input_dict = dictionary
        self.windowed_data = self.input_dict["data_body"]["windows_list"]
        self.step = self.input_dict["data_body"]["window_step"]
        self.len = self.input_dict["data_body"]["window_len"]
        self.data = self.input_dict["data_body"]["elected_data"]
        self.lables = self.input_dict["data_body"]["raw_lables"]
        self.win_len = self.input_dict["data_body"]["window_len"]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Areas detector: Start transforming...")
        self._areas_analysis()
        self._print_logs(f"{get_current_time()} Areas detector: Transforming finished!")

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

    def _areas_analysis(self) -> None:
        
        def get_distance_1(data: list, number: int) -> float:
                temp_list = []
                for dataset in data:
                    temp_list.append(dataset[number])
                return abs(max(temp_list) - min(temp_list))
        
        def get_distance(data: list, number: int) -> float:
                if data[0][number] >= data[1][number]: return abs(data[0][number] - data[1][number])
                else: return abs(data[1][number] - data[0][number])

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
                    if data[0][i] - data[1][i] >= 0: state = 1
                    else: state = -1
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
            max_val = max(odd_new_predicts) + max(odd_new_predicts)/10
            #odd_new_predicts = list(map(lambda x: max_val if x > q_95 else 0, score_diff))
            reshaped_data = preprocessing.normalize([np.array(odd_new_predicts)]).flatten()
            reshaped_data = self.NormalizeData(np.array(odd_new_predicts))
            self.output_list.append(reshaped_data.tolist())
            #self.output_list.append(odd_new_predicts)


    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
