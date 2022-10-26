from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector
from cases.anomaly_detection.clear_architecture.utils.get_time import time_now


class WindowCutter(AbstractDetector):
    """
    Window cutter.
        input format: dict with "data" and "labels" fields
        output: the same dict but with additional windows_list and labels for it
    """
    def __init__(self, window_len, window_step):
        super().__init__(name="Window Cutter", operation="window cutting")
        self.window_len = window_len
        self.window_step = window_step
        self.output_window_list = []

    def input_data(self, input_dict: dict) -> None:
        self._print_logs(f"{time_now()} {self.name}: Data read!")
        self.input_dict = input_dict
        self.data = self.input_dict["data_body"]["elected_data"]

    def output_data(self) -> dict:
        self.input_dict["data_body"]["windows_list"] = self.output_window_list
        self.input_dict["data_body"]["window_len"] = self.window_len
        self.input_dict["data_body"]["window_step"] = self.window_step
        return self.input_dict

    def _do_analysis(self) -> None:
        """
        Cut data to windows
        :return: none
        """
        for data in self.data:
            windows = self._cut_ts_to_windows(data)
            self.output_window_list.append(windows)

    def _cut_ts_to_windows(self, ts: list) -> list:
        start_idx = 0
        end_idx = len(ts[0]) - self.window_len
        temp_windows_list = []
        for i in range(start_idx, end_idx, self.window_step):
            temp_window = []
            for _ in ts:
                temp_window.append([])
            for j in range(i, i + self.window_len):
                for t in range(len(temp_window)):
                    temp_window[t].append(ts[t][j])
            temp_windows_list.append(temp_window)

        return temp_windows_list
