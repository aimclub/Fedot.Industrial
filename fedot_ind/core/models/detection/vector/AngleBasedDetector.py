from typing import Mapping

from scipy import spatial

from fedot_ind.core.models.detection.abstract_objects.AbstractDataOperation import AbstractDataOperation
from fedot_ind.core.models.detection.abstract_objects.FileObject import FileObject
from fedot_ind.core.models.detection.utils.get_time import time_now
from fedot_ind.core.models.detection.utils.math_utils import NormalizeData

"""

input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
from typing import List


class Window:
    def __init__(self,
                 start: int,
                 end: int,
                 data_ts: Mapping[str, List[float]]
                 ) -> None:
        self.start: int = start
        self.end: int = end
        self.data_ts: Mapping[str, List[float]] = data_ts
        self.vector_result: float = 0

    def get_vector_predict(self) -> None:
        point_array: List[List[float]] = []
        keys = list(self.data_ts.keys())
        for i in range(0, self.end - self.start):
            point = []
            for key in keys:
                point.append(self.data_ts[key][i])
            point_array.append(point)

        cosine_array = []
        # last_point = point_array[-1]
        first_point = point_array[0]
        for i in range(0, len(point_array)):
            current_point = point_array[i]
            res = spatial.distance.cosine(current_point, first_point)
            cosine_array.append(res)
        avg = sum(cosine_array) / len(cosine_array)
        self.vector_result = avg  # ** 2


class AngleBasedDetector(AbstractDataOperation):

    def __init__(self, window_len: int = 100,
                 step: int = None,
                 detector_name: str = "Angle Based Detector"):
        self.window_len: list = window_len
        self.name: str = detector_name
        self.print_logs: bool = True
        if step is None:
            self.step: int = int(window_len / 4)
        else:
            self.step: int = step
        self.windows: List[Window] = []

    def set_settings(self):
        self.output_predicts: list = []
        self._print_logs("Settings was set.")

    def load_data(self, data_object) -> None:
        self._print_logs("Data read!")
        self.data_object: FileObject = data_object

    def run_operation(self) -> None:
        self._print_logs("Start detection...")
        self._run_detector()
        self._print_logs("Detection finished!")

    def return_new_data(self):
        return self.data_object

    def _split_data_to_windows(self,
                               ts: Mapping[str, List[float]],
                               len_of_data: int,
                               keys: List[str]
                               ) -> List[Window]:
        """
        This method split time series into windows

        Args:
            ts (Mapping[str, List[float]]): data
            len_of_data (int): len of data in <ts>
            keys (List[str]): keys of <ts> dict

        Returns:
            List[Window]: windows
        """
        start_idx: int = 0
        end_idx: int = len_of_data - self.window_len
        temp_window_list: List[Window] = []
        for i in range(start_idx, end_idx, self.step):
            temp_window_data: Mapping[str, List[float]] = {}
            for key in keys:
                temp_window_data[key] = []
                for j in range(i, i + self.window_len):
                    temp_window_data[key].append(ts[key][j])
            temp_window_list.append(Window(i, i + self.window_len, temp_window_data))
        return temp_window_list

    def _turn_windows_to_ts(self, windows: List[Window], demanded_len) -> List[float]:
        """
        This method turn list of windows back to ts suitable for visualization

        Args:
            windows (List[Window]): list of windows
            demanded_len (_type_): len of ts

        Returns:
            List[float]: time series
        """
        output_ts: List[float] = []
        for window in windows:
            for _ in range(0, self.step):
                output_ts.append(window.vector_result)
        for _ in range(len(output_ts), demanded_len):
            output_ts.append(0)
        return output_ts

    def _run_detector(self) -> None:
        self.windows = []
        self.windows = self._split_data_to_windows(
            self.data_object.time_series_data,
            self.data_object.get_len_of_dataset(),
            list(self.data_object.time_series_data.keys())
        )

        for j in range(len(self.windows)):
            self.windows[j].get_vector_predict()
        self.data_object.test_vector_ts = \
            self._turn_windows_to_ts(
                self.windows, self.data_object.get_len_of_dataset()
            )
        self.data_object.test_vector_ts = NormalizeData(
            self.data_object.test_vector_ts
        )

    def _print_logs(self, log_message: str) -> None:
        if self.print_logs:
            print(f"---[{time_now()}] {self.name}: {log_message}")
