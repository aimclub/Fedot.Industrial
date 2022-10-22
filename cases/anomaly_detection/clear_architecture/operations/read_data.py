import os
from datetime import datetime

from tqdm import tqdm

from cases.anomaly_detection.clear_architecture.operations.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.clear_architecture.utils.data_types import CLEAR_DATA
from cases.anomaly_detection.clear_architecture.utils.get_time import time_now
from cases.anomaly_detection.data.label_reader import read_labels_csv_from_file


class DataReader(AbstractDataOperation):
    """
    Data Reader class.
        data_type: <type>,
        data_body: <main data>,
        data_flags: <{dict of flags}>
    """

    def __init__(self, data_path, labels_path):
        super().__init__(name="Data Reader", operation="data reading")
        self.labels_path = labels_path

        self.labels = None
        self.refined_labels = None
        self.refined_data = None
        self.labels_for_show = None
        self.all_labels = None
        self.data_path = data_path

    def input_data(self, args=None) -> None:
        self._print_logs(f"{time_now()} {self.name}: Start reading...")

    def run(self) -> None:
        self.all_labels = ["N", "DIST", "Xu", "Yu", "Zu", "Xd", "Yd", "Zd", "Vu", "Vd", "LAT", "LNG", "Time", "Depth"]
        self._print_logs(f"{time_now()} {self.name}: Try to read labels...")
        self.labels = read_labels_csv_from_file(self.labels_path)
        self._print_logs(f"{time_now()} {self.name}: Labels read successful!")
        self._print_logs(f"{time_now()} {self.name}: Try to read data...")
        self.refined_data, self.labels_for_show, self.refined_labels = self._read_data_csv_in_folder(self.data_path)
        self._print_logs(f"{time_now()} {self.name}: Data is ready!")

    def output_data(self) -> dict:
        return dict(data_type=CLEAR_DATA,
                    data_body={"raw_labels": self.refined_labels,
                               "raw_data": self.refined_data,
                               "raw_columns": self.all_labels,
                               "labels_for_show": self.labels_for_show
                               },
                    data_flags={})

    def _read_data_csv_in_folder(self, path_to_folder: str):
        files = []
        self._print_logs(f"{time_now()} {self.name}: Creating file list...")
        for file in os.listdir(path_to_folder):
            if file.endswith(".CSV"):
                files.append(os.path.join(self.data_path, file))
        self._print_logs(f"{time_now()} {self.name}: File list created! {len(files)} files found!")
        formatted_data = []
        formatted_labels = []
        labels_for_show = []
        for file in files:
            data = self._read_data_csv_from_file(file)
            # get file's labels
            label = []
            filename = os.path.splitext(os.path.basename(file))[0]
            for i in range(len(self.labels)):
                if self.labels[i][0] == filename:
                    label = self.labels[i][1]
            temp_label_arr_for_work = [0] * len(data)
            temp_label_arr_fow_show = [0] * len(data)
            approx_count = 30
            for i, line in enumerate(data):
                for label_line in label:
                    if int(label_line[0]) - approx_count <= line[0] <= int(label_line[1]) + approx_count:
                        temp_label_arr_fow_show[i] = 0.5
                        temp_label_arr_for_work[i] = 1
            formatted_data.append(data)
            labels_for_show.append(temp_label_arr_fow_show)
            formatted_labels.append(temp_label_arr_for_work)

        return formatted_data, labels_for_show, formatted_labels

    def _read_data_csv_from_file(self, filename: str) -> list:
        with open(filename, 'r', encoding="iso-8859-1") as file:
            lines = file.readlines()
        list_to_save = []
        for i in tqdm(range(2, len(lines), 1)):  # len(lines)-2
            v = bytes(lines[i], encoding="iso-8859-1")
            good_str = v.decode("iso-8859-1").replace('\x00', '').replace(' ', '')
            arr = good_str.split(";")
            temp_arr = []
            for j in range(0, len(self.all_labels)):
                if j == 12:
                    temp_arr.append(datetime.strptime(arr[j], "%H:%M:%S").time())
                elif j == 0:
                    temp_arr.append(int(arr[j]))
                else:
                    temp_arr.append(float(arr[j]))
            list_to_save.append(temp_arr)
        return list_to_save

    def _do_analysis(self) -> None:
        pass
