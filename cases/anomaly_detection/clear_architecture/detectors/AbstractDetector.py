import numpy as np
from sklearn.metrics import f1_score

from cases.anomaly_detection.clear_architecture.utils.get_time import get_current_time as time_now
from cases.anomaly_detection.clear_architecture.utils.settings_args import SettingsArgs


class AbstractDetector:
    args: SettingsArgs
    data: list = []
    output_list: list = []
    windowed_data: list = []
    filtering: bool = False

    def __init__(self, name: str = 'DetectorName'):
        self.input_dict = None
        self.name = name

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{time_now()} {self.name}: settings was set.")
        self._print_logs(f"{time_now()} {self.name}: Visualize = {self.args.visualize}")
        self._print_logs(f"{time_now()} {self.name}: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{time_now()} {self.name} detector: Data read!")
        self.input_dict = dictionary
        self.windowed_data = self.input_dict["data_body"]["windows_list"]
        self.step = self.input_dict["data_body"]["window_step"]
        self.len = self.input_dict["data_body"]["window_len"]
        self.data = self.input_dict["data_body"]["elected_data"]
        self.labels = self.input_dict["data_body"]["raw_labels"]
        self.win_len = self.input_dict["data_body"]["window_len"]

    def run(self) -> None:
        self._print_logs(f"{time_now()} {self.name}: Start transforming...")
        self._analysis()
        self._print_logs(f"{time_now()} {self.name}: Transforming finished!")

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
                score.append(f1_score(self.labels[i], self.output_list[i], average='macro'))
            print("-------------------------------------")
            main_score = sum(score) / len(score)
            print("Average predict:")
            print(main_score)
            print("-------------------------------------")
        return self.input_dict

    def _analysis(self) -> None:
        pass

    @staticmethod
    def _make_vector(point_1: list, point_2: list):
        if len(point_1) != len(point_2):
            raise ValueError("Vectors has to be the same len!")
        vector = []
        for i in range(len(point_1)):
            vector.append(point_2[i] - point_1[i])
        return vector

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
