from sklearn.metrics import f1_score
from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
import numpy as np
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation

from TSAnomalyDetector.utils.math_utils import NormalizeData, NormalizeDataForDetectors
from TSAnomalyDetector.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, PREDICTIONS_FOR_VISUALIZATION, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, QUANTILE_PREDICTIONS
from TSAnomalyDetector.abstract_classes.DataObject import DataObject
from TSAnomalyDetector.abstract_classes.SuspiciouslyZone import SuspiciouslyZone
from TSAnomalyDetector.abstract_classes.AnomalyZone import AnomalyZone
from scipy.signal import savgol_filter
from TSAnomalyDetector.utils.math_utils import NormalizeData, Cut_data, NormalizeData_1


"""

input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
from typing import List, Type

class SimpleDetector(AbstractDataOperation):
    """
    Detector that combines three metrics:
        A - area of zone,
        L - length of zone,
        D - max distance of zones
    by weights:
        D * W1 + L * W2 + A * W3

    
    """
    args: SettingsArgs

    def __init__(self, zero_percentage = 0.5,
            ignore_first: bool = False,
            ignore_last: bool = False,
            detector_name: str = "Default name"):
        self.ignore_last = ignore_last
        self.ignore_first = ignore_first
        self.target_percentage = zero_percentage
        self.detector_name = detector_name

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.output_predicts: list = []
        self._print_logs(f"{get_current_time()} Simple detector: settings was set.")
        self._print_logs(f"{get_current_time()} Simple detector: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Simple detector: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Simple detector: Data read!")
        self.data_object = data_object
        #self.input_dict[DATA_BODY][ELECTED_DATA]
        #self.lables = self.input_dict[DATA_BODY][RAW_LABLES]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Simple detector: Start detection...")
        self._vector_analysis()
        self._print_logs(f"{get_current_time()} Simple detector: Detection finished!")

    def output_data(self) -> DataObject:
        return self.data_object

    def get_percentage_of_zero(self, time_series: list) -> float:
        """

        Args:
            time_series (list): _description_

        Returns:
            float: percentage of 0 in list
        """
        return time_series.count(0)/len(time_series)

    def _vector_analysis(self) -> None:   
        self.output_predicts = []
        self.output_quantile_predicts = []
        self.output_quantile_predicts_for_show = []
        
        self.target_error = 0.1
        self.low_thresh = self.target_percentage - self.target_error
        self.high_thresh = self.target_percentage + self.target_error
        for filename in self.data_object.get_list_of_files():
            current_thresh = 0.3
            step = 0.005
            self.data_object.ensambled_prediction[filename] = []


            temp_data_exp = savgol_filter(self.data_object.distance_ts[filename], 87, 1) 
            temp_data_exp = savgol_filter(temp_data_exp, 31, 1)
            temp_data_exp = savgol_filter(temp_data_exp, 87, 1) 
            temp_data_exp = savgol_filter(temp_data_exp, 31, 1)
            temp_data_exp = savgol_filter(temp_data_exp, 87, 1) 
            temp_data_exp = savgol_filter(temp_data_exp, 31, 1)
            counter = 0

            while True:
                counter+=1
                temp_reshaped_data = Cut_data(temp_data_exp, current_thresh)
                if self.low_thresh <= self.get_percentage_of_zero(temp_reshaped_data)<=self.high_thresh or counter >= 200:
                    temp_data_exp = temp_reshaped_data
                    break
                else:
                    if self.get_percentage_of_zero(temp_reshaped_data) > self.target_percentage:
                        current_thresh-=step
                    else:
                        current_thresh+=step
            self.data_object.distance_ts_for_exp[filename] = temp_data_exp



            areas_list, len_list, areas_size_list, distances_list = self._get_areas_of_data(temp_data_exp)
            scaled_predict = [0] * self.data_object.get_len_of_dataset(filename)
            ensambled_prediction: List[Type[AnomalyZone]] = []

            if len(areas_list) and len(len_list) and len(distances_list):
            
                for area in areas_list:
                    if area[1] - area[0] > 200:
                        ensambled_prediction.append(AnomalyZone(area[0], area[1]))

            self.data_object.ensambled_prediction[filename] = ensambled_prediction
            #self.output_predicts.append(scaled_predict)
            #self.output_quantile_predicts.append(predict_out)
            #self.output_quantile_predicts_for_show.append(predict_for_show)
        self.data_object.copy_parts_of_datasets_in_anomaly_zones()



    def _get_areas_of_data(self, data: list):
        """

        Args:
            data (list): _description_
        """
        def get_distance(data: list) -> float:
            return abs(max(data) - min(data))
        # returns three lists
        # 1 - we are in, 0 - we are out, 2 - start
        state_of_scaning: int = 2
        counter_for_areas = 0
        areas_list = []
        len_list = []
        areas_sizes_list = []
        max_list = []
        max_dist = 0
        area = 0
        start = 0
        end = 0
        flag_1 = False
        flag_2 = False
        
        for i in range(len(data)):
            if state_of_scaning == 2:
                if data[i] != 0:
                    state_of_scaning = 1
                    max_dist = max(max_dist, data[i])
                    start = i
                else:
                    state_of_scaning = 0
                    start = i
                continue

            if state_of_scaning == 1:
                if data[i] == 0: 
                    state_of_scaning = 0
                    end = i
                    areas_list.append([start, end])
                    max_list.append(max_dist)
                    len_list.append(counter_for_areas)
                    areas_sizes_list.append(area)
                    flag_1 = False
                    flag_2 = False
                    counter_for_areas = 0
                    max_dist = 0
                    start = i
                    area = 0
                else:
                    counter_for_areas += 1

                    area += data[i]
                    max_dist = max(max_dist, data[i])  
                continue
            if state_of_scaning == 0:
                if data[i] != 0:  
                    start = i
                    state_of_scaning = 1
                    counter_for_areas = 1
                    area = 0
                    max_dist = max(max_dist, data[i])
                continue
        if state_of_scaning == 1:
            areas_list.append([start, len(data)])
            max_list.append(max_dist)
            len_list.append(counter_for_areas)
            areas_sizes_list.append(area)
        if self.ignore_first:
            del areas_list[0]
            del max_list[0]
            del len_list[0]
            del areas_sizes_list[0]
        if self.ignore_last:
            del areas_list[-1]
            del max_list[-1]
            del len_list[-1]
            del areas_sizes_list[-1]

        return areas_list, len_list, areas_sizes_list, max_list


    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
