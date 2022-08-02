from datetime import datetime
import sys
import os

from tqdm import tqdm
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../../"))
from anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time
from anomaly_detection.clear_architecture.settings_args \
    import SettingsArgs
from anomaly_detection.clear_architecture.constants.data_types \
    import CLEAR_DATA
"""
init
input_data
run
output_data
set_settings



{
    data_type: <type>,
    data_body: <main data>,
    data_flags: <{dict of flags}>
}
"""
class DataReader:
    args: SettingsArgs
    data_path: str
    lables_path: str

    def __init__(self, data_path, lables_path):
        self.data_path = data_path
        self.lables_path = lables_path

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Data loader: settings was set.")
        self._print_logs(f"{get_current_time()} Data loader: Visualisate = {self.args.visualisate}")
        self._print_logs(f"{get_current_time()} Data loader: Print logs = {self.args.print_logs}")

    def input_data(self) -> None:
        self._print_logs(f"{get_current_time()} Data loader: Start reading...")


    def run(self) -> None:
        self.all_lables = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Time", "Depth"]
        self._print_logs(f"{get_current_time()} Data loader: Try to read lables...")
        self.lables = self._read_lables_csv_from_file(self.lables_path)
        self._print_logs(f"{get_current_time()} Data loader: Lables read successful!")
        self._print_logs(f"{get_current_time()} Data loader: Try to read data...")
        self.refied_data, self.refined_lables = self._read_data_csv_in_folder(self.data_path)
        self._print_logs(f"{get_current_time()} Data loader: Data is ready!")

    def output_data(self) -> dict:
        return {
            "data_type": CLEAR_DATA,
            "data_body": {
                "raw_lables": self.refined_lables,
                "raw_data": self.refied_data, 
                "raw_columns": self.all_lables
                },
            "data_flags": {}
        }

    def _read_lables_csv_from_file(self, filename: str) -> list:
        temp_list = [[], [], [], [], []]
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for i in tqdm(range(0, len(lines), 1)): #len(lines)-2
            temp_line = lines[i].strip().split(";")
            for j in range(len(temp_list)):
                temp_list[j].append(temp_line[j])
        out_list = []
        current_filename = ""
        temp_element = []
        temp_anomalies_list = []
        for i in range(len(temp_list[0])):
            if current_filename == "":
                current_filename = temp_list[0][i].strip()
                temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
                continue
            if current_filename!=temp_list[0][i].strip():
                temp_element = [current_filename, temp_anomalies_list]
                out_list.append(temp_element)
                current_filename = temp_list[0][i].strip()
                temp_anomalies_list = []
                temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
            else:
                temp_anomalies_list.append([temp_list[1][i], temp_list[2][i], temp_list[3][i], temp_list[4][i]])
        temp_element = [current_filename, temp_anomalies_list]
        out_list.append(temp_element)
        return out_list


    def _read_data_csv_in_folder(self, path_to_folder: str):
        files = []
        self._print_logs(f"{get_current_time()} Data loader: Creating file list...")
        for file in os.listdir(path_to_folder):
            if file.endswith(".CSV"):
                files.append(os.path.join(self.data_path, file))
        self._print_logs(f"{get_current_time()} Data loader: File list created! {len(files)} files found!")
        formatted_data = []
        fromatted_lables = []
        for file in files:
            data = self._read_data_csv_from_file(file)
            #get file's lables
            lable = []
            filename = os.path.splitext(os.path.basename(file))[0]
            for i in range(len(self.lables)):
                if self.lables[i][0] == filename: 
                    lable = self.lables[i][1]
            temp_lable_arr = [0] * len(data)
            approx_count = 30
            for i, line in enumerate(data):
                for lable_line in lable:
                    if int(lable_line[0]) - approx_count <= line[0]<= int(lable_line[1])+approx_count:
                        temp_lable_arr[i] = 1
            formatted_data.append(data)
            fromatted_lables.append(temp_lable_arr)
        return formatted_data, fromatted_lables

    def _read_data_csv_from_file(self, filename: str) -> list:
        with open(filename, 'r', encoding="iso-8859-1") as file:
            lines = file.readlines()
        list_to_save = []
        for i in tqdm(range(2, len(lines), 1)): #len(lines)-2
            v = bytes(lines[i], encoding="iso-8859-1")
            good_str = v.decode("iso-8859-1").replace('\x00','').replace(' ','')
            arr = good_str.split(";")
            temp_arr = []
            for j in range(0, len(self.all_lables)):
                if j == 12:
                    temp_arr.append(datetime.strptime(arr[j], "%H:%M:%S").time())
                elif j == 0:
                    temp_arr.append(int(arr[j]))
                else:   
                    temp_arr.append(float(arr[j]))
            list_to_save.append(temp_arr)
        return list_to_save

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)