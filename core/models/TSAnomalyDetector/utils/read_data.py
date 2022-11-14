from datetime import datetime
import os

from tqdm import tqdm

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


class DataReader(AbstractDataOperation):
    args: SettingsArgs
    data_path: str
    lables_path: str

    def __init__(self, data_path, lables_path, min_anomaly_index: int = 17):
        self.data_path = data_path
        self.lables_path = lables_path
        self.min_anomaly_index = min_anomaly_index

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Data loader: settings was set.")
        self._print_logs(f"{get_current_time()} Data loader: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Data loader: Print logs = {self.args.print_logs}")

    def input_data(self) -> None:
        self._print_logs(f"{get_current_time()} Data loader: Start reading...")


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Data loader: Try to read lables...")
        self.lables = self._read_lables_csv_from_file(self.lables_path)
        self._print_logs(f"{get_current_time()} Data loader: Lables read successful!")
        self._print_logs(f"{get_current_time()} Data loader: Try to read data...")
        self.main_data_dict, self.main_lables_dict = self._read_data_csv_in_folder(self.data_path)
        self._print_logs(f"{get_current_time()} Data loader: Data is ready!")

    def output_data(self) -> DataObject:
        if len(list(self.main_data_dict.keys())) != len(list(self.main_lables_dict.keys())):
            raise ValueError("Lens of lables and data isn't the same!")
        output_object = DataObject(self.main_data_dict, self.main_lables_dict)
        #output_object.main_data_dict = self.main_data_dict
        #output_object.main_lables_dict = self.main_lables_dict

        #print(output_object.get_common_columns())
        #print(output_object.get_list_of_files())
        #key = output_object.get_list_of_files()[0]
        #print(output_object.get_lables_for_visualization(key))
        return output_object

    def _read_lables_csv_from_file(self, filename: str) -> dict:
        """
        Read lables from pre formatted file in format:
            <file name>, <anomaly start>, <anomaly end>, <index>, <Description>
        Args:
            filename (str): full path to lables file

        Returns:
            set: {<file name>, <list of anomalies>}
        """
        # in lables 5 columns - redo after
        # <file name>, <anomaly start>, <anomaly end>, <ЛП>, <Description>
        # maybe will add another fields in future
        temp_list_from_anomaly_file = [[], [], [], [], []]
        # file read
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for i in tqdm(range(0, len(lines), 1)): #len(lines)-2
            temp_line = lines[i].strip().split(";")
            for j in range(len(temp_list_from_anomaly_file)):
                if j == 0:
                    temp_list_from_anomaly_file[j].append(f"{temp_line[j]}.CSV")
                elif j == 1 or j == 2:
                    temp_list_from_anomaly_file[j].append(int(temp_line[j]))
                elif j == 3:
                    try:
                        temp_list_from_anomaly_file[j].append(int(temp_line[j]))
                    except:
                        temp_list_from_anomaly_file[j].append(-1)
                else:
                    temp_list_from_anomaly_file[j].append(temp_line[j])

        lables_dict = {}

        for i in range(len(temp_list_from_anomaly_file[0])):
            temp_lable = [temp_list_from_anomaly_file[1][i], temp_list_from_anomaly_file[2][i], temp_list_from_anomaly_file[3][i], temp_list_from_anomaly_file[4][i]]
            if temp_list_from_anomaly_file[3][i] == "": temp_list_from_anomaly_file[3][i] = -1
            else: temp_list_from_anomaly_file[3][i] = int(temp_list_from_anomaly_file[3][i])
            if temp_list_from_anomaly_file[3][i] != -1 or temp_list_from_anomaly_file[3][i] > self.min_anomaly_index:
                if temp_list_from_anomaly_file[0][i] in lables_dict:
                    lables_dict[temp_list_from_anomaly_file[0][i]].append(temp_lable)
                else:
                    lables_dict[temp_list_from_anomaly_file[0][i]] = [temp_lable]

        return lables_dict


    def _read_data_csv_in_folder(self, path_to_folder: str):
        """
        Method read all .CSV files in folder

        Args:
            path_to_folder (str): full path to folder

        Returns:
            list: formatted_data
            list: lables_for_show
            list: fromatted_lables
        """
        self.full_files_paths = []
        self.files_names = []
        self._print_logs(f"{get_current_time()} Data loader: Creating file list...")
        for file in os.listdir(path_to_folder):
            if file.endswith(".CSV"):
                self.full_files_paths.append(os.path.join(self.data_path, file))
                self.files_names.append(file)
        self._print_logs(f"{get_current_time()} Data loader: File list created! {len(self.files_names)} files found!")
        main_data_dict = {}
        main_lables_dict = {}
        for i, filepath in enumerate(self.full_files_paths):
            main_data_dict[self.files_names[i]] = self._read_data_csv_from_file(filepath)
            if self.files_names[i] in self.lables:
                main_lables_dict[self.files_names[i]] = self.lables[self.files_names[i]]
            else:
                main_lables_dict[self.files_names[i]] = []

        # this if for one particular dataset...
        
        return main_data_dict, main_lables_dict

    def _read_data_csv_from_file(self, filename: str) -> list:
        """
        This method read data from one csv file.
        There is several important params...

            start - all files have bad first two lines, so we have to read from minimal 3 line, 
            but also many files have bad start data, so I read file from 100-400 line.
            end - many files have bad end data, so param sets where before the end of file have we to stop read files
        Args:
            filename (str): full path to csv file

        Returns:
            list: list of data in horizontal format: [[field1, field2, .. , fieldn], [], .., []]
        """
        self._print_logs(f"{get_current_time()} Data loader: Read data from {filename}")
        with open(filename, 'r', encoding="iso-8859-1") as file:
            lines = file.readlines()
        # decode
        line_in_bytes_format = bytes(lines[1], encoding=DATA_FORMAT)
        # format and strip
        good_str = line_in_bytes_format.decode(DATA_FORMAT).replace('\x00','').replace(' ','')
        good_str = good_str.rstrip('\r\n')
        # split
        column_lables = good_str.split(DELIMITER)

        dict_to_save = {}
        for key in column_lables:
            dict_to_save[key] = []
        # ++ -------------------------------- ++
        # First line is some odd file name or full file name that looks terrible 
        # 'cause of troubles with encoding. Anyway, first line isn't useful
        # Second line contains lables of columns. Looks on - start line is 0!
        start = 2
        if start < 0: raise ValueError("Can't read first two lines in files...")
        # ++ -------------------------------- ++
        for i in tqdm(range(start, len(lines), 1)): #len(lines)-2
            # decode
            line_in_bytes_format = bytes(lines[i], encoding=DATA_FORMAT)
            # get rid of spaces and other special symbols
            good_str = line_in_bytes_format.decode(DATA_FORMAT).replace('\x00','').replace(' ','')
            good_str = good_str.rstrip('\r\n')
            # split
            line_in_array_form = good_str.split(DELIMITER)
            for j, key in enumerate(column_lables):
                if key == "Time":
                    if j<len(line_in_array_form):
                        dict_to_save[key].append(datetime.strptime(line_in_array_form[j], "%H:%M:%S").time())
                    else: dict_to_save[key].append(0)
                elif key == "Comment":
                    # 0 is number of row
                    if j<len(line_in_array_form):
                        dict_to_save[key].append(line_in_array_form[j])
                    else: dict_to_save[key].append(0)
                elif key == "N":
                    # 0 is number of row
                    dict_to_save[key].append(int(line_in_array_form[j]))
                else:   
                    # the rest is float values
                    if j<len(line_in_array_form)-1:
                        dict_to_save[key].append(float(line_in_array_form[j]))
                    else: dict_to_save[key].append(0)
        # check that all lists are the same length
        temp_len_array = []
        for key in column_lables:
            temp_len_array.append(len(dict_to_save[key]))
        if len(set(temp_len_array)) != 1:
            raise ValueError("File read wasn't successful! Lengths of readed arrays aren't the same!")
        return dict_to_save

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)