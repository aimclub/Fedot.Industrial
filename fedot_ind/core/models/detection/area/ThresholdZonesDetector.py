from fedot_ind.core.models.detection.utils.get_time \
    import time_now
from fedot_ind.core.models.detection.abstract_objects.AbstractDataOperation import AbstractDataOperation

from fedot_ind.core.models.detection.utils.math_utils import NormalizeData, Cut_data

from fedot_ind.core.models.detection.abstract_objects.AnomalyZone import AnomalyZone
from fedot_ind.core.models.detection.abstract_objects.FileObject import FileObject

"""

input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
from typing import List, Type

class ThresholdZonesDetector(AbstractDataOperation):
    """
    This detector cut ts up to demanded percentage of 0 in it and 
    make detection of anomaly zones that could be used in future

    Args:
        AbstractDetector (_type_): _description_
    """
    def __init__(self, zero_percentage = 0.5,
            ignore_first: bool = False,
            ignore_last: bool = False,
            detector_name: str = "Threshold Zones Detector"):
        self.ignore_last: bool = ignore_last
        self.ignore_first: bool = ignore_first
        self.target_percentage: float = zero_percentage
        self.name: str = detector_name
        self.print_logs: bool = True

    def set_settings(self, print_logs: bool = True):
        self.print_logs = print_logs
        self._print_logs("Settings was set.")
        self._print_logs(f"Print logs = {print_logs}")

    def load_data(self, file_object: FileObject) -> None:
        self._print_logs("Data read!")
        self.file_object: FileObject = file_object
        #self.input_dict[DATA_BODY][ELECTED_DATA]
        #self.lables = self.input_dict[DATA_BODY][RAW_LABLES]

    def run_operation(self) -> None:
        self._print_logs("Start detection...")
        self._vector_analysis()
        self._print_logs("Detection finished!")

    def return_new_data(self) -> FileObject:
        return self.file_object

    def get_percentage_of_zero(self, time_series: List[float]) -> float:
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
        
        self.target_error = 0.05
        self.low_thresh = self.target_percentage - self.target_error
        self.high_thresh = self.target_percentage + self.target_error
        self._print_logs(f"Predict on {self.file_object.filename}...")

        current_thresh = 0.3
        step = 0.005
        self.file_object.anomalies_list = []
        # -------------------------------------
        temp_data_exp = self.file_object.test_data

        counter = 0
        self._print_logs(f"Predict on {self.file_object.filename}, attempt to transform data to {self.target_percentage} of zero...")

        while True:
            counter+=1
            temp_reshaped_data = Cut_data(temp_data_exp, current_thresh)
            if self.low_thresh <= self.get_percentage_of_zero(temp_reshaped_data)<=self.high_thresh or counter >= 200:
                temp_data_exp = temp_reshaped_data
                self.file_object.threshold = current_thresh
                self._print_logs(f"Predict on {self.file_object.filename}, succes! Current threshold: {current_thresh}")
                break
            else:
                if self.get_percentage_of_zero(temp_reshaped_data) > self.target_percentage:
                    current_thresh-=step
                else:
                    current_thresh+=step
        self.file_object.additional_transformed_average_absolute_deviation = \
            temp_data_exp



        areas_list, len_list, distances_list = self._get_areas_of_data(temp_data_exp)
        ensambled_prediction: List[Type[AnomalyZone]] = []
        self._print_logs(f"Predict on {self.file_object.filename}, filtering...")

        if len(areas_list) and len(len_list) and len(distances_list):
            for area in areas_list:
                if area[1] - area[0] >= 100:
                    ensambled_prediction.append(AnomalyZone(area[0], area[1]))

        self.file_object.anomalies_list = ensambled_prediction

        for j in range(self.file_object.get_len_of_dataset()):
            if self.file_object.additional_transformed_average_absolute_deviation[j] != 0:
                self.file_object.additional_transformed_average_absolute_deviation[j] = \
                    self.file_object.additional_transformed_average_absolute_deviation[j] - current_thresh
        self.file_object.additional_transformed_average_absolute_deviation = \
            NormalizeData(self.file_object.additional_transformed_average_absolute_deviation)

        self.file_object.copy_parts_of_datasets_in_anomaly_zones()



    def _get_areas_of_data(self, data: list):
        """

        Args:
            data (list): _description_
        """
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

        return areas_list, len_list, max_list


    def _print_logs(self, log_message: str) -> None:
        if self.print_logs:
            print(f"---[{time_now()}] {self.name}: {log_message}")