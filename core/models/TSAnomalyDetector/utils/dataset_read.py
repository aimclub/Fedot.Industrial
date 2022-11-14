from datetime import datetime
import os
import json

from tqdm import tqdm
from typing import List, Type
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation

from TSAnomalyDetector.utils.get_time \
    import get_current_time
from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.constants.data_types \
    import CLEAR_DATA
from TSAnomalyDetector.constants.data_read_constants \
    import DELIMITER, DATA_FORMAT, APPROX_COUNT
from TSAnomalyDetector.abstract_classes.DataObject import DataObject
from TSAnomalyDetector.abstract_classes.AnomalyZone import AnomalyZone
from scipy.signal import savgol_filter



class DatasetReader(AbstractDataOperation):
    args: SettingsArgs
    data_path: str
    lables_path: str

    def __init__(self, files_path: str):
        self.files_path = files_path

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Dataset loader: settings was set.")
        self._print_logs(f"{get_current_time()} Dataset loader: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Dataset loader: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self.data_object = data_object
        self._print_logs(f"{get_current_time()} Dataset loader: Start reading...")


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Dataset loader: Try to read lables...")
        self.read_dataset()
        self._print_logs(f"{get_current_time()} Dataset loader: Lables read successful!")
        self._print_logs(f"{get_current_time()} Dataset loader: Data is ready!")

    def output_data(self) -> DataObject:
        return self.data_object
    

    def read_dataset(self) -> List[Type[AnomalyZone]]:

        from os import walk

        f = []
        filenames = next(walk(self.files_path), (None, None, []))[2]  # [] if no file    print(filenames)
        anomalies = []
        for i in tqdm(range(len(filenames))): 
            with open(os.path.join(self.files_path, filenames[i]), "r") as stream:
                anomalies.append(json.load(stream))
        #print(filenames)
        temp_list: List[Type[AnomalyZone]] = []
        for i in range(len(anomalies)):
            temp_list.append(AnomalyZone())
            """
            for key in anomalies[i].keys():
                if key != "type" and key != "heaviness" and key != "comment":
                    try:
                        temp_data_exp = savgol_filter(anomalies[i][key], 87, 1) 
                        temp_data_exp = savgol_filter(temp_data_exp, 31, 1) 
                        temp_list[-1].data[key] = temp_data_exp
                    except:
                        temp_list[-1].data[key] = anomalies[i][key]
            temp_list[-1].data = anomalies[i]
            temp_list[-1].anomaly_class = anomalies[i]["type"]
            temp_list[-1].heaviness = anomalies[i]["heaviness"]
            temp_list[-1].comment = anomalies[i]["comment"]
            """
            for key in anomalies[i].keys():
                if key == "vector":
                    temp_list[-1].features = anomalies[i][key]
                if key == "heaviness":
                    temp_list[-1].heaviness = anomalies[i][key]
                if key == "type":
                    temp_list[-1].anomaly_class = anomalies[i][key]
                if key == "comment":
                    temp_list[-1].comment = anomalies[i][key]
        
        self.data_object.database = temp_list
        print(len(filenames))
        print(len(anomalies))
