from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time
from typing import List, Type
from sklearn.metrics import f1_score
import pickle
from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, LABLES_FOR_VISUALIZATION, CLUSTERIZATOR, \
        MAIN_METRIC, ENSAMBLED_PREDICTION, ENSAMBLED_PREDICTION_FOR__VISUALIZATION, CLEAR_CLUSTERED_PREDICT, MAIN_METRIC_CLUST
from cases.anomaly_detection.abstract_classes.DataObject import DataObject
from cases.anomaly_detection.abstract_classes.SuspiciouslyZone import SuspiciouslyZone
from cases.anomaly_detection.abstract_classes.AnomalyZone import AnomalyZone

"""

    
"""
class Ensembling(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []
    def __init__(self, quantile: float = 0.95, threshold: bool = False) -> None:
        self.quantile = quantile
        self.threshold = threshold


    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Metrics: settings was set.")
        self._print_logs(f"{get_current_time()} Metrics: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Metrics: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Metrics: Data read!")
        self.data_object = data_object
        #self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        #self.lables = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        #self.lables_for_metrics = self.input_dict[DATA_BODY][RAW_LABLES]
        #self.predicts_q = self.input_dict[DATA_BODY][DETECTIONS][RAW_PREDICTIONS]


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Metrics: Loading metrics...")
        self._Metrics()
        self._print_logs(f"{get_current_time()} Metrics: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

    def _update_or_check_zone(self, zones_list: List[Type[SuspiciouslyZone]], zone: SuspiciouslyZone) -> SuspiciouslyZone:
        
        def check_zone(checked_zone: SuspiciouslyZone, cur_zone: SuspiciouslyZone) -> SuspiciouslyZone:
            """
            False if zones don't intersect
            new zone if intersect
            """
            # if zones the same
            if checked_zone.get_start() == cur_zone.get_start() and \
                checked_zone.get_end() == cur_zone.get_end(): 
                checked_zone.metric = max(checked_zone.get_metric(), cur_zone.get_metric())
                return checked_zone, True
            # if zones intersect
            if cur_zone.get_start() <= checked_zone.get_start() <= cur_zone.get_end() \
                or cur_zone.get_start() <= checked_zone.get_end() <= cur_zone.get_end():
                checked_zone.start = min(cur_zone.get_start(), checked_zone.get_start())
                checked_zone.end = max(cur_zone.get_end(), checked_zone.get_end())
                checked_zone.metric = max(checked_zone.get_metric(), cur_zone.get_metric())
                return checked_zone, True
            return checked_zone, False
        # --------------------------------------

        for i in range(len(zones_list)):
            zone, flag = check_zone(zone, zones_list[i])
            if flag:
                zones_list[i] = zone
                return zones_list, False
        return zones_list, True

    def _Metrics(self) -> None:
        self.ensambled_predict  = []
        self.ensambled_predict_for_show = []
        scores = []
        PREDICT_IN_LINEAR_FORM = "linear predict"
        AREAS_LIST = "areas list"

        for filename in self.data_object.get_list_of_files():
            temp_predicts = []
            temp_areas = []
            for key in list(self.data_object.predicts_dict[filename].keys()):
                temp_predicts.append(self.data_object.predicts_dict[filename][key][PREDICT_IN_LINEAR_FORM])
                temp_areas.append(self.data_object.predicts_dict[filename][key][AREAS_LIST])
            ensambled_prediction: List[Type[AnomalyZone]] = []
            temp_ensamble_predict_areas: List[Type[SuspiciouslyZone]]  = []
            if len(temp_areas) == 0: raise ValueError("Use at least one detector before ensambling!")
            if len(temp_areas[0]) == 0: 
                self.data_object.ensambled_prediction[filename] = ensambled_prediction
                continue
            if len(temp_areas) == 1: temp_ensamble_predict_areas = temp_areas[0]
            else:
                for zone_list in temp_areas:
                    for zone in zone_list:
                        temp_ensamble_predict_areas, result = self._update_or_check_zone(temp_ensamble_predict_areas, zone)
                        if result:
                            temp_ensamble_predict_areas.append(zone)
                        if temp_ensamble_predict_areas[-1].get_start() > self.data_object.get_len_of_dataset(filename) or \
                            temp_ensamble_predict_areas[-1].get_end() > self.data_object.get_len_of_dataset(filename):
                            print( temp_ensamble_predict_areas[-1])

            values = []
            for area in temp_ensamble_predict_areas:
                if not area.get_metric() in values: values.append(area.get_metric())
            values.sort()
            length = len(values)
            q_index = int(length * self.quantile)
            if q_index > len(values):
                q_index-=1
            quantile = values[q_index]
            if self.threshold: quantile = self.quantile
            for area in temp_ensamble_predict_areas:
                if area.get_metric() >= quantile and area.get_end() - area.get_start() > 10:
                    ensambled_prediction.append(AnomalyZone(area.get_start(), area.get_end()))

            self.data_object.ensambled_prediction[filename] = ensambled_prediction

            #temp_ensamble_predict = [0] * self.data_object.get_len_of_dataset(filename)
            #for predict in temp_predicts:
            #    for j, element in enumerate(predict):
            #        temp_ensamble_predict[j] = max(element, temp_ensamble_predict[j])

            #values = []
            #for value in temp_ensamble_predict:
            #    if not value in values: values.append(value)
            #values.sort()
            #length = len(values)
            #q_index = int(length * self.quantile)
            #if q_index > len(values):
            #    q_index-=1
            #quantile = values[q_index]
            #quantile = self.quantile
            #if self.threshold: quantile = self.quantile
            #predict_out = list(map(lambda x: 1 if x >= quantile else 0, temp_ensamble_predict))
            # predict for show is a little bigger than 1 to be more useful
            #predict_for_show = list(map(lambda x: 0.6 if x >= quantile else 0, temp_ensamble_predict))
            #score = f1_score(self.data_object.get_lables_for_metric(filename), predict_out, average='macro')
            #score = self.custom_metric(
            #    self.data_object.get_lables_for_metric(filename), 
            #    predict_out, 
            #    0.3)
            #scores.append(score)
            #self._print_logs(f"{get_current_time()} Metrics: {scores[-1]}")
            
            #self._print_logs(f"{get_current_time()} Metrics: {score}")
            
            #scores.append(scores[-1])
            #self.ensambled_predict.append(predict_out)
            #self.ensambled_predict_for_show.append(predict_for_show)
        self._print_logs(f"{get_current_time()} Metrics: ----------------------------------------")
        #self.main_score = sum(scores) / len(scores)
        self._print_logs(f"{get_current_time()} Metrics: Average predict:")
        #self._print_logs(f"{get_current_time()} Metrics: {self.main_score}")
        self._print_logs(f"{get_current_time()} Metrics: -------------------------------------")
        self.data_object.copy_parts_of_datasets_in_anomaly_zones()
    
    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)


    def custom_metric(self, lables: list, predict: list, percentage: float) -> float:
        if len(lables) != len(predict): raise ValueError("Arrays aren't the same len!")
        counter_of_lables = 0
        counter_of_detections = 0
        current_counter_pred = 0
        current_counter_lables = 0
        for i in range(len(lables)):
            if lables[i] == 1:
                current_counter_lables += 1
                if predict[i] == 1:
                    current_counter_pred += 1
            else:
                if current_counter_lables != 0:
                    counter_of_lables+=1
                    if  current_counter_pred / current_counter_lables>= percentage:
                        counter_of_detections += 1
                    current_counter_pred = 0
                    current_counter_lables = 0
                else:
                    current_counter_pred = 0
                    current_counter_lables = 0
        if lables[-1] == 1:
            counter_of_lables+=1
            if  current_counter_pred / current_counter_lables >= percentage:
                counter_of_detections += 1            
        if counter_of_lables == 0: return 1
        if counter_of_lables > 0:
            return counter_of_detections / counter_of_lables
        return 0