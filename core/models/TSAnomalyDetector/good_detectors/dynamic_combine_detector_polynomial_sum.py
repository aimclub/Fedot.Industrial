from sklearn.metrics import f1_score
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time
import numpy as np
from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation

from cases.anomaly_detection.utils.math_utils import NormalizeData, NormalizeDataForDetectors
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, PREDICTIONS_FOR_VISUALIZATION, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, QUANTILE_PREDICTIONS
from cases.anomaly_detection.abstract_classes.DataObject import DataObject
from cases.anomaly_detection.abstract_classes.SuspiciouslyZone import SuspiciouslyZone
"""

input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class DynamicCombineDetectorPolySum(AbstractDataOperation):
    """
    Detector that combines three metrics:
        A - area of zone,
        L - length of zone,
        D - max distance of zones
    by weights:
        D * W1 + L * W2 + A * W3

    
    """
    args: SettingsArgs

    def __init__(self, quantile: float = 0.95, 
            distance_weight: float = 1, 
            len_weight: float = 1,
            areas_weight: float = 1,
            ignore_first: bool = False,
            ignore_last: bool = False,
            detector_name: str = "Default name"):
        self.ignore_last = ignore_last
        self.ignore_first = ignore_first
        self.quantile = quantile
        self.distance_weight = distance_weight
        self.areas_weight = areas_weight
        self.len_weight = len_weight
        self.detector_name = detector_name

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.output_predicts: list = []
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: settings was set.")
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Data read!")
        self.data_object = data_object
        #self.input_dict[DATA_BODY][ELECTED_DATA]
        #self.lables = self.input_dict[DATA_BODY][RAW_LABLES]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Start detection...")
        self._vector_analysis()
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Detection finished!")

    def output_data(self) -> DataObject:
        return self.data_object

    def _vector_analysis(self) -> None:   
        self.output_predicts = []
        self.output_quantile_predicts = []
        self.output_quantile_predicts_for_show = []

        PREDICT_IN_LINEAR_FORM = "linear predict"
        AREAS_LIST = "areas list"


        

        for filename in self.data_object.get_list_of_files():
            self.data_object.predicts_dict[filename][self.detector_name] = {}
            data, _ = self.data_object.get_transformed_data(filename)
            
            #predict_out = [0] * self.data_object.get_len_of_dataset(filename)
            #predict_for_show = [0] * self.data_object.get_len_of_dataset(filename)
            #reshaped_data = [0] * self.data_object.get_len_of_dataset(filename)
            #print(self.data_object.get_len_of_dataset(filename))
            areas_list, len_list, areas_size_list, distances_list = self._get_areas_of_data(data)
            scaled_predict = [0] * self.data_object.get_len_of_dataset(filename)
            temp_areas_list = []
            if len(areas_list) and len(len_list) and len(distances_list):
                len_list = NormalizeDataForDetectors(np.array(len_list)).tolist()
                areas_size_list = NormalizeDataForDetectors(np.array(areas_size_list)).tolist()
                distances_list = NormalizeDataForDetectors(np.array(distances_list)).tolist()

                max_areas_list = []
                for item_number in range(len(areas_size_list)):
                    max_areas_list.append(
                        (len_list[item_number] ** self.len_weight) +
                        (areas_size_list[item_number] ** self.areas_weight) +
                        (distances_list[item_number] ** self.distance_weight) 
                    )
                
                if len(max_areas_list):
                    max_areas_list_scaled = NormalizeDataForDetectors(np.array(max_areas_list)).tolist()
                
                    for j, window_idxs in enumerate(areas_list):
                        for i in range(window_idxs[0], window_idxs[1]):
                            scaled_predict[i] = max_areas_list_scaled[j]
                    for j, window_idxs in enumerate(areas_list):
                        temp_areas_list.append(SuspiciouslyZone(window_idxs[0], window_idxs[1], max_areas_list_scaled[j]))
                    #reshaped_data = NormalizeDataForDetectors(np.array(odd_scaled_predict)).tolist()
                    values = []
                    for value in scaled_predict:
                        if not value in values: values.append(value)
                    values.sort()
                    length = len(values)
                    quantile = values[int(length * self.quantile)]
                    #quantile = self.quantile
                    predict_out = list(map(lambda x: 1 if x >= quantile else 0, scaled_predict))
                    # predict for show is a little bigger than 1 to be more useful
                    predict_for_show = list(map(lambda x: 1.1 if x >= quantile else 0, scaled_predict))
            self.data_object.predicts_dict[filename][self.detector_name][PREDICT_IN_LINEAR_FORM] = scaled_predict
            self.data_object.predicts_dict[filename][self.detector_name][AREAS_LIST] = temp_areas_list
            #self.output_predicts.append(scaled_predict)
            #self.output_quantile_predicts.append(predict_out)
            #self.output_quantile_predicts_for_show.append(predict_for_show)

    def _get_areas_of_data(self, data: dict):
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
        keys = list(data.keys())
        if len(keys) != 2: raise ValueError("There need only two ts' for detector!")
        
        for i in range(len(data[keys[0]])):
            data_frame = [data[keys[0]][i], data[keys[1]][i]]
            if state_of_scaning == 2:
                if abs(data[keys[0]][i] - data[keys[1]][i])!= 0:
                    state_of_scaning = 1
                    start = i
                else:
                    state_of_scaning = 0
                    start = i
                continue

            if state_of_scaning == 1:
                if abs(data[keys[0]][i] - data[keys[1]][i]) == 0: 
                    state_of_scaning = 0
                    end = i
                    if flag_1 and flag_2:
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
                    area += get_distance(data_frame)
                    max_dist = max(max_dist, get_distance(data_frame))
                    if data[keys[0]][i] != 0:
                        flag_1 = True
                    if data[keys[1]][i] != 0:
                        flag_2 = True    
                continue
            if state_of_scaning == 0:
                if abs(data[keys[0]][i] - data[keys[1]][i]) != 0:  
                    start = i
                    state_of_scaning = 1
                    counter_for_areas = 1
                    area = 0
                continue
        if state_of_scaning == 1:
            areas_list.append([start, len(data[keys[0]])])
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
