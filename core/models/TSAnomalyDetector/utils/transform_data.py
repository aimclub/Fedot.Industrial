
import numpy as np
from tqdm import tqdm
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation
from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
from statistics import mean
from scipy.signal import savgol_filter

from TSAnomalyDetector.utils.math_utils import NormalizeData, Cut_data, NormalizeData_1
from TSAnomalyDetector.constants.current_data_const import \
    DATA_BODY, RAW_DATA, COLUMNS_LABLES, TRANSFORMED_DATA
from TSAnomalyDetector.abstract_classes.DataObject import DataObject


from statistics import median
"""

input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class DataTransformer(AbstractDataOperation):
    """
    We read data in horizontal format:
        [[field1, field2, .. , fieldn], [], .., []]
    This element transform this data in following format:
        [[field1, field1, .. , field1], [field2, field2, .. , field2], .., []]

    Returns:
        _type_: _description_
    """
    args: SettingsArgs
    
    def __init__(self, smooth_count: int = 10, min_thresh: float = 0.078) -> None:
        self.smooth_count = smooth_count
        self.target_percentage = min_thresh

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.transformed_data = []
        self.state = 0
        #self.target_percentage = 0.5
        self.target_error = 0.1
        self.low_thresh = self.target_percentage - self.target_error
        self.high_thresh = self.target_percentage + self.target_error
        self._print_logs(f"{get_current_time()} Data transformator: settings was set.")
        self._print_logs(f"{get_current_time()} Data transformator: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Data transformator: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Data transformator: Data read!")
        self.data_object = data_object

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Data transformator: Start transforming...")
        self._all_data_transorm()
        self._print_logs(f"{get_current_time()} Data transformator: Transforming finished!")

    def output_data(self) -> DataObject:
        return self.data_object

    def _all_data_transorm(self) -> None:
        for filename in self.data_object.get_list_of_files():
            self.data_object.transformed_data[filename], self.data_object.settings_for_data[filename] = self._data_transform(filename)

    def _data_transform(self, filename: str) -> dict:
        transformed_dict, settings_for_transformed_dict = self.data_object.get_elected_raw_data(filename)

        SMOOTH_COUNT = "smooth count"
        THRESHOLD = "threshold"
        STATE = "state"


        # prepare data
        average_mean = 0
        state = 0
        start = 500
        end = 500

        for key in self.data_object.main_data_dict[filename].keys():
            try:
                temp_data_exp = savgol_filter(self.data_object.main_data_dict[filename][key], 87, 1) 
                temp_data_exp = savgol_filter(temp_data_exp, 31, 1)
                temp_data_exp = savgol_filter(temp_data_exp, 87, 1) 
                temp_data_exp = savgol_filter(temp_data_exp, 31, 1)
                if mean(temp_data_exp) < average_mean:
                    mean_distance = average_mean - mean(temp_data_exp)
                    for j in range(len(temp_data_exp)):
                        temp_data_exp[j] = temp_data_exp[j] + abs(mean_distance)
                else:
                    mean_distance = average_mean - mean(temp_data_exp)
                    for j in range(len(temp_data_exp)):
                        temp_data_exp[j] = temp_data_exp[j] - abs(mean_distance)
                if len(temp_data_exp) > start:
                    for j in range(0, start):
                        temp_data_exp[j] = average_mean
                if len(temp_data_exp) > end:
                    for j in range(len(temp_data_exp)-end, len(temp_data_exp)):
                        temp_data_exp[j] = average_mean
                temp_data_exp = NormalizeData(np.array(temp_data_exp))
                self.data_object.experimented_data[filename][key] = temp_data_exp
            except:
                self.data_object.experimented_data[filename][key] = self.data_object.main_data_dict[filename][key]

        keys = ["Xu", "Yu", "Zu", "Vu", "Xd", "Yd", "Zd", "Vd"]
        self.data_object.min_ts[filename]
        
        for i in range(self.data_object.get_len_of_dataset(filename)):
            current_min = 100000
            current_max = -100000
            for key in keys:
                current_min = min(self.data_object.experimented_data[filename][key][i], current_min)
                current_max = max(self.data_object.experimented_data[filename][key][i], current_max)
            self.data_object.min_ts[filename].append(current_min)
            self.data_object.max_ts[filename].append(current_max)
        
        self.data_object.min_ts[filename] = NormalizeData(np.array(self.data_object.min_ts[filename]))
        self.data_object.max_ts[filename] = NormalizeData(np.array(self.data_object.max_ts[filename]))
        if len(temp_data_exp) > start:
            for j in range(0, start):
                self.data_object.min_ts[filename][j] = 0
        if len(temp_data_exp) > end:
            for j in range(len(temp_data_exp)-end, len(temp_data_exp)):
                self.data_object.min_ts[filename][j] = 0

        #self.data_object.min_ts[filename] = Cut_data(self.data_object.min_ts[filename], current_thresh)
        #self.data_object.max_ts[filename] = Cut_data(self.data_object.max_ts[filename], current_thresh)

        for i in range(self.data_object.get_len_of_dataset(filename)):
            self.data_object.distance_ts[filename].append(abs(self.data_object.max_ts[filename][i]-self.data_object.min_ts[filename][i]))
            self.data_object.distance_ts_for_exp[filename].append(abs(self.data_object.max_ts[filename][i]-self.data_object.min_ts[filename][i]))
        if len(temp_data_exp) > start:
            for j in range(0, start):
                self.data_object.distance_ts[filename][j] = 0
                self.data_object.distance_ts_for_exp[filename][j] = 0
        if len(temp_data_exp) > end:
            for j in range(len(temp_data_exp)-end, len(temp_data_exp)):
                self.data_object.distance_ts[filename][j] = 0
                self.data_object.distance_ts_for_exp[filename][j] = 0

        
        
        for i, key in enumerate(list(transformed_dict.keys())):
            counter = 0
            temp_data = transformed_dict[key]
            current_thresh = 0.3
            step = 0.005
            reshaped_data = NormalizeData(np.array(temp_data))
            # remember about one smooth!
            reshaped_data = savgol_filter(reshaped_data, 87, 1) 
            reshaped_data = savgol_filter(reshaped_data, 31, 1) 

            if mean(reshaped_data) < average_mean:
                mean_distance = average_mean - mean(reshaped_data)
                for j in range(len(reshaped_data)):
                    reshaped_data[j] = reshaped_data[j] + abs(mean_distance)
            else:
                mean_distance = average_mean - mean(reshaped_data)
                for j in range(len(reshaped_data)):
                    reshaped_data[j] = reshaped_data[j] - abs(mean_distance)

            reshaped_data = self.TransformDataToOneSide(reshaped_data, state)
            for _ in range(self.smooth_count):
                reshaped_data = savgol_filter(reshaped_data, 87, 1) 
                reshaped_data = savgol_filter(reshaped_data, 31, 1)
            if self.target_percentage:
                while True:
                    counter+=1
                    temp_reshaped_data = Cut_data(reshaped_data, current_thresh)
                    if self.low_thresh <= self.get_percentage_of_zero(temp_reshaped_data)<=self.high_thresh or counter >= 100:
                        reshaped_data = temp_reshaped_data
                        transformed_dict[key] = reshaped_data
                        break
                    else:
                        if self.get_percentage_of_zero(temp_data) > self.target_percentage:
                            current_thresh-=step
                        else:
                            current_thresh+=step
            settings_for_transformed_dict[key][SMOOTH_COUNT] = self.smooth_count
            settings_for_transformed_dict[key][THRESHOLD] = current_thresh
            settings_for_transformed_dict[key][STATE] = state
            if state == 0: state = 1
            else: state = 0

        return transformed_dict, settings_for_transformed_dict


    def TransformDataToOneSide(self, data, state):
        out_data = []
        if state == 0:
            state = 1
            for element in data:
                if element < 0: out_data.append(-1 * element)
                else: out_data.append(element)
        else:
            state = 0
            for element in data:
                if element > 0: out_data.append(-1 * element)
                else: out_data.append(element)
        return out_data

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)

    def get_percentage_of_zero(self, time_series: list) -> float:
        """

        Args:
            time_series (list): _description_

        Returns:
            float: percentage of 0 in list
        """
        return time_series.count(0)/len(time_series)