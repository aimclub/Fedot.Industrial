"""
Data structures! Important!

Raw data readed from files. Can't be changed!
self.data_from_files =
    {
        <file name from self.file_list>: {
            <time series column lables>: [45.1, 45.2, .., 45.5]
        }  
    }
self.lables =
    {
        <file name from self.file_list>: {
            <anomaly_number>: {
                <zone>: [start, end],
                <heaviness>: <int>
                <description>: <str>
            }
        }  
    }

self.database =
    [
        { 
            <data>: {<time series column lables>: [time series]},
            <heaviness>: <int>(0-3)
            <description>: <str> ???
        }  
    ]

"""
from typing import Type, List, Dict
from TSAnomalyDetector.abstract_classes.AnomalyZone import AnomalyZone
from TSAnomalyDetector.abstract_classes.SuspiciouslyZone import SuspiciouslyZone

class DataObject:
    def __init__(self, main_data_dict: dict, main_lables_dict: dict) -> None:
        self.main_data_dict: dict = main_data_dict
        self.main_lables_dict: dict = main_lables_dict
        self.database: List[Type[AnomalyZone]] = None
        self.allowed_lables = []
        self.current_elected_columns: list = []
        self.current_state = None # Do it later
        # ????
        # -------------------
        self.clusterizator = None
        self.reducer = None
        # -------------------
        self.transformed_data: dict = {}
        self.settings_for_data: dict = {}

        self.dataset: List[Type[AnomalyZone]] = []

        self.clusters = []

        self.predicts_dict: dict = {}
        self.experimented_data = {}
        self.ensambled_prediction: dict[Type[AnomalyZone]] = {}
        self.min_ts = {}
        self.max_ts = {}
        self.distance_ts = {}
        self.distance_ts_for_exp = {}
        for key in self.get_list_of_files():
            self.settings_for_data[key] = {}
            self.min_ts[key] = []
            self.max_ts[key] = []
            self.distance_ts_for_exp[key] = []
            self.distance_ts[key] = []
            self.transformed_data[key] = {}
            self.experimented_data[key] = {}
            self.predicts_dict[key] = {}
            self.ensambled_prediction[key] = {}
        PREDICT_IN_LINEAR_FORM = "linear predict"
        AREAS_LIST = "areas list"
        self.clusters_adress: list = []


    def get_list_of_files(self) -> list:
        return list(self.main_data_dict.keys())
    
    def check_columns_for_correctness(self, columns: list) -> bool:
        """
        Method allows to check columns for being common for every dataset in class
    

        Args:
            columns (list): _description_

        Returns:
            bool: _description_
        """
        common_columns = []
        for key in self.main_data_dict.keys():
            if len(common_columns) == 0:
                common_columns = list(self.main_data_dict[key].keys())
            else:
                temp_columns = list(self.main_data_dict[key].keys())
                for column in common_columns:
                    if column not in temp_columns:
                        common_columns.remove(column)
        for column in columns:
            if column not in common_columns:
                return False
        return True

    def get_common_columns(self) -> list:
        common_columns = []
        for key in self.main_data_dict.keys():
            if len(common_columns) == 0:
                common_columns = list(self.main_data_dict[key].keys())
            else:
                temp_columns = list(self.main_data_dict[key].keys())
                for column in common_columns:
                    if column not in temp_columns:
                        common_columns.remove(column)
        return common_columns

    def get_lables_list(self, filename: str, min_heavines: int = 12) -> list:
        if len(self.main_lables_dict[filename]):
            temp_list = []
            for predict in self.main_lables_dict[filename]:
                if predict[2] > min_heavines:
                    temp_list.append(predict)
            return temp_list
        else:
            return False

    def make_last_predict(self) -> None:
        i = 3
        if False:
            for filename in self.get_list_of_files():
                for number in range(len(self.ensambled_prediction[filename])):
                    if self.ensambled_prediction[filename][number].predicted_type == i:
                        if self.ensambled_prediction[filename][number].dataset_type != 0:
                            if self.ensambled_prediction[filename][number].dataset_type == 2:
                                self.ensambled_prediction[filename][number].predicted_type = 2
                        else:
                            self.ensambled_prediction[filename][number].predicted_type = 2
                    else:
                        if self.ensambled_prediction[filename][number].dataset_type == i:
                            self.ensambled_prediction[filename][number].predicted_type = i
                    
            i = 2
            for filename in self.get_list_of_files():
                for number in range(len(self.ensambled_prediction[filename])):
                    if self.ensambled_prediction[filename][number].predicted_type == i:
                        if self.ensambled_prediction[filename][number].dataset_type != 0:
                            self.ensambled_prediction[filename][number].predicted_type = i
                        else:
                            self.ensambled_prediction[filename][number].predicted_type = 2
                    else:
                        if self.ensambled_prediction[filename][number].dataset_type == i and self.ensambled_prediction[filename][number].predicted_type != 3:
                            self.ensambled_prediction[filename][number].predicted_type = i
            for filename in self.get_list_of_files():
                for number in range(len(self.ensambled_prediction[filename])):
                    if self.ensambled_prediction[filename][number].dataset_type == 3:
                        self.ensambled_prediction[filename][number].predicted_type = 3
        else:
            for filename in self.get_list_of_files():
                for number in range(len(self.ensambled_prediction[filename])):
                    #self.ensambled_prediction[filename][number].predicted_type = self.ensambled_prediction[filename][number].dataset_type
                    self.ensambled_prediction[filename][number].predicted_type = 0

                    

    def get_predicts_list(self, filename: str, type_number: int) -> list:
        if not len(self.ensambled_prediction[filename]):
            return False
        else: 
            if type_number == -1:
                out_list = []
                for predict in self.ensambled_prediction[filename]:
                    out_list.append(predict)
            else:
                out_list = []
                for predict in self.ensambled_prediction[filename]:
                    if predict.predicted_type == type_number: # and predict.heaviness >9: 
                    #if predict.cluster_type == type_number: 
                        out_list.append(predict)
            return out_list

    def get_predict_for_metrics(self, filename: str) -> list:
        lables_for_show = [0] * self.get_len_of_dataset(filename)
        # lables is really bad, so we make them a little bigger
        approx_count = 0 #APPROX_COUNT

        if len(self.ensambled_prediction[filename]) == 0:
            return lables_for_show
        else:
            for zone in self.ensambled_prediction[filename]:
                # <anomaly start>, <anomaly end>, <index>, <Description>
                if zone.predicted_type == 2 or zone.predicted_type == 3:
                    for j in range(zone.get_start(), zone.get_end(), 1):
                        if 0 <= j < self.get_len_of_dataset(filename):
                            lables_for_show[j] = 1 #0.5
        return lables_for_show


    def get_lables_for_metrics(self, filename: str, min_index: int = 12) -> list:
        lables_for_show = [0] * self.get_len_of_dataset(filename)
        # lables is really bad, so we make them a little bigger
        approx_count = 0 #APPROX_COUNT

        if len(self.main_lables_dict[filename]) == 0:
            return lables_for_show
        else:
            for line in self.main_lables_dict[filename]:
                # <anomaly start>, <anomaly end>, <index>, <Description>
                if line[2] >= 2:
                    for j in range(line[0]-approx_count, line[1]+approx_count, 1):
                        if 0 <= j < self.get_len_of_dataset(filename) and min_index < line[2]:
                            lables_for_show[j] = 1 #0.5
        return lables_for_show
    
    def get_ts_of_predict(self, filename: str, type: int) -> list:
        ts_for_show = [0] * self.get_len_of_dataset(filename)
        # lables is really bad, so we make them a little bigger
        approx_count = 0 #APPROX_COUNT
        
        if len(self.ensambled_prediction[filename]) == 0:
            return ts_for_show
        else:
            for zone in self.ensambled_prediction[filename]:
                if zone.predicted_type == type and zone.dataset_type != 0:
                    for j in range(zone.get_start(), zone.get_end(), 1):
                        ts_for_show[j] = 0.5 #0.5
                else:
                    if zone.dataset_type == type:
                        for j in range(zone.get_start(), zone.get_end(), 1):
                            ts_for_show[j] = 0.5 #0.5
        return ts_for_show


    def get_len_of_dataset(self, filename: str) -> int:
        keys = list(self.main_data_dict[filename].keys())
        return len(self.main_data_dict[filename][keys[0]])

    def get_current_elected_columns(self) -> list:
        if len(self.current_elected_columns) == 0:
            return False
        else: 
            return self.current_elected_columns

    def get_elected_raw_data(self, filename: str) -> list:
        start = 500
        end = 400
        if len(self.current_elected_columns) == 0:
            return False
        else: 
            SMOOTH_COUNT = "smooth count"
            THRESHOLD = "threshold"
            STATE = "state"
            out_dict: dict = {}
            out_empty_settings_dict: dict = {}
            for key in self.current_elected_columns:
                out_dict[key] = self.main_data_dict[filename][key]
                average = sum(out_dict[key]) / len(out_dict[key])
                for i in range(0, start):
                    out_dict[key][i] = average
                for i in range(len(out_dict[key])-end, len(out_dict[key])):
                    out_dict[key][i] = average   
                out_empty_settings_dict[key] = {}
                out_empty_settings_dict[key][SMOOTH_COUNT] = None
                out_empty_settings_dict[key][THRESHOLD] = None
                out_empty_settings_dict[key][STATE] = None
            return out_dict, out_empty_settings_dict
    
    def get_transformed_data(self, filename: str) -> dict:
        return self.transformed_data[filename], self.settings_for_data[filename]

    def copy_parts_of_datasets_in_anomaly_zones(self) -> None:
        for key in self.get_list_of_files():
            for i in range(len(self.ensambled_prediction[key])):
                for column in list(self.main_data_dict[key].keys()):
                    self.ensambled_prediction[key][i].data[column] = self.main_data_dict[key][column][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]
                self.ensambled_prediction[key][i].min_data = self.min_ts[key][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]
                self.ensambled_prediction[key][i].max_data = self.max_ts[key][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]
                
                self.ensambled_prediction[key][i].distance_data = self.distance_ts[key][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]
                self.ensambled_prediction[key][i].distance_data_trans = self.distance_ts_for_exp[key][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]
        keys = ["Xu", "Yu", "Zu", "Vu"]

        for key in self.get_list_of_files():
            for i in range(len(self.ensambled_prediction[key])):
                for column in list(self.main_data_dict[key].keys()):
                    self.ensambled_prediction[key][i].data_for_dataset[column] = self.experimented_data[key][column][self.ensambled_prediction[key][i].get_start():self.ensambled_prediction[key][i].get_end()]

