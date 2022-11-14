from sklearn.metrics import f1_score
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time
import numpy as np
from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation

from cases.anomaly_detection.utils.math_utils import NormalizeData

from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, PREDICTIONS_FOR_VISUALIZATION, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, QUANTILE_PREDICTIONS
"""
╔═ DATA_TYPE: flag of status of pypeline, currently unused
╠═ DATA_BODY: {
║   ╠═ RAW_LABLES: lables in format: [x, x, .., x] where x 0 or 1
║   ╠═ LABLES_FOR_VISUALISATION: lables in format: [x, x, .., x] where x 0 or n, n could be any
║   ╠═ RAW_DATA: data from files, in horizonal format: [[field1_1, field2_1, .. , fieldN_1], [field1_2, field_2_2, .. , fieldN_2], .., []]
║   ╠═ COLUMNS_LABLES: columns names
║   ╠═ TRANSFORMED_DATA: data in vertical format : [[field1_1, field1_2, .. , field1_N], [field2_1, field2_2, .. , field2_N], .., []]
║   ╠═ ELECTED_LABLES: from transformer data we have to elect some data by this lables
║   ╠═ ELECTED_DATA: data, choosed by ELECTED_LABLES
║   ╠═ LIST_OF_WINDOWS: list of windows in which we cut data by STEP_OF_WINDOWS and LEN_OF_WINDOWS
║   ╠═ STEP_OF_WINDOWS: step of windows
║   ╠═ LEN_OF_WINDOWS: length of windows
║   ╚═ DETECTIONS
║       ║ --------------------------    
║       ║ ++ We are here: ++
║       ╠═ RAW_PREDICTIONS: raw predictions from detector, stacked!
║       ╠═ QUANTILE_PREDICTIONS: predictions from each detector filtered by respected quantile
║       ╠═ PREDICTIONS_FOR_VISUALIZATION
║       ║ --------------------------
║       ╠═ ENSAMBLED_PREDICTION: prediction made from all predictions
║       ╠═ MAIN_METRIC: main metric of ensambling
╚═ DATA_FLAGS: currently unused



input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class DynamicCombineDetectorLinear(AbstractDataOperation):
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
            ignore_last: bool = False):
        self.ignore_last = ignore_last
        self.ignore_first = ignore_first
        self.quantile = quantile
        self.distance_weight = distance_weight
        self.areas_weight = areas_weight
        self.len_weight = len_weight

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self.output_predicts: list = []
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: settings was set.")
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        self.lables = self.input_dict[DATA_BODY][RAW_LABLES]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Start detection...")
        self._vector_analysis()
        self._print_logs(f"{get_current_time()} Dynamic min-max detector: Detection finished!")

    def output_data(self) -> dict:
        if DETECTIONS in self.input_dict[DATA_BODY]:
            previous_predict = self.input_dict[DATA_BODY][DETECTIONS][RAW_PREDICTIONS]
            for i in range(len(self.output_predicts)):
                self.output_predicts[i] = [self.output_predicts[i]]
            for i in range(len(self.output_predicts)):
                for j in range(len(previous_predict[i])):
                    self.output_predicts[i].append(previous_predict[i][j])
            previous_quantile_predict = self.input_dict[DATA_BODY][DETECTIONS][QUANTILE_PREDICTIONS]
            for i in range(len(self.output_quantile_predicts)):
                self.output_quantile_predicts[i] = [self.output_quantile_predicts[i]]
            for i in range(len(self.output_quantile_predicts)):
                for j in range(len(previous_quantile_predict[i])):
                    self.output_quantile_predicts[i].append(previous_quantile_predict[i][j])
            previous_predict_for_show = self.input_dict[DATA_BODY][DETECTIONS][PREDICTIONS_FOR_VISUALIZATION]
            for i in range(len(self.output_quantile_predicts_for_show)):
                self.output_quantile_predicts_for_show[i] = [self.output_quantile_predicts_for_show[i]]
            for i in range(len(self.output_quantile_predicts_for_show)):
                for j in range(len(previous_predict_for_show[i])):
                    self.output_quantile_predicts_for_show[i].append(previous_predict_for_show[i][j])
        else:
            for i in range(len(self.output_predicts)):
                self.output_predicts[i] = [self.output_predicts[i]]
            for i in range(len(self.output_quantile_predicts)):
                self.output_quantile_predicts[i] = [self.output_quantile_predicts[i]]
            for i in range(len(self.output_quantile_predicts_for_show)):
                self.output_quantile_predicts_for_show[i] = [self.output_quantile_predicts_for_show[i]]
        
        self.input_dict[DATA_BODY][DETECTIONS] = \
            {
                RAW_PREDICTIONS: self.output_predicts,
                QUANTILE_PREDICTIONS:  self.output_quantile_predicts,
                PREDICTIONS_FOR_VISUALIZATION: self.output_quantile_predicts_for_show
            }
        return self.input_dict

    def _vector_analysis(self) -> None:   
        self.output_predicts = []
        self.output_quantile_predicts = []
        self.output_quantile_predicts_for_show = []


        for data in self.data:
            predict_out = [0] * len(data[0])
            predict_for_show = [0] * len(data[0])
            reshaped_data = [0] * len(data[0])
            areas_list, len_list, areas_size_list, distances_list = self._get_areas_of_data(data)
            if len(areas_list) and len(len_list) and len(distances_list):
                len_list = NormalizeData(np.array(len_list)).tolist()
                areas_size_list = NormalizeData(np.array(areas_size_list)).tolist()
                distances_list = NormalizeData(np.array(distances_list)).tolist()

                max_areas_list = []
                for item_number in range(len(areas_size_list)):
                    max_areas_list.append(
                        (len_list[item_number] * self.len_weight) +
                        (areas_size_list[item_number] * self.areas_weight) +
                        (distances_list[item_number] * self.distance_weight) 
                    )
                
                if len(max_areas_list):
                    max_areas_list_scaled = NormalizeData(np.array(max_areas_list)).tolist()
                
                    odd_scaled_predict = [0] * len(data[0])
                    for j, window_idxs in enumerate(areas_list):
                        if j == 0 and self.ignore_first: continue
                        if j == len(areas_list)-1 and self.ignore_last: continue
                        for i in range(window_idxs[0], window_idxs[1]):
                            odd_scaled_predict[i] = max_areas_list_scaled[j]

                    #reshaped_data = NormalizeData(np.array(odd_scaled_predict)).tolist()
                    reshaped_data = odd_scaled_predict
                    values = []
                    for value in reshaped_data:
                        if not value in values: values.append(value)
                    values.sort()
                    length = len(values)
                    quantile = values[int(length * self.quantile)]
                    #quantile = self.quantile
                    predict_out = list(map(lambda x: 1 if x >= quantile else 0, reshaped_data))
                    # predict for show is a little bigger than 1 to be more useful
                    predict_for_show = list(map(lambda x: 1.1 if x >= quantile else 0, reshaped_data))
            self.output_predicts.append(reshaped_data)
            self.output_quantile_predicts.append(predict_out)
            self.output_quantile_predicts_for_show.append(predict_for_show)

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
        for i in range(len(data[0])):
            data_frame = [data[0][i], data[1][i]]
            if state_of_scaning == 2:
                if abs(data[0][i] - data[1][i])!= 0:
                    state_of_scaning = 1
                    start = i
                else:
                    state_of_scaning = 0
                    start = i
                continue

            if state_of_scaning == 1:
                if abs(data[0][i] - data[1][i]) == 0: 
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
                    if data[0][i] != 0:
                        flag_1 = True
                    if data[1][i] != 0:
                        flag_2 = True    
                continue
            if state_of_scaning == 0:
                if abs(data[0][i] - data[1][i]) != 0:  
                    start = i
                    state_of_scaning = 1
                    counter_for_areas = 1
                    area = 0
                continue
        if state_of_scaning == 1:
            areas_list.append([start, len(data[0])])
            max_list.append(max_dist)
            len_list.append(counter_for_areas)
            areas_sizes_list.append(area)
        return areas_list, len_list, areas_sizes_list, max_list


    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
