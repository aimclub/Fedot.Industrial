
from tqdm import tqdm
from anomaly_detection.clear_architecture.settings_args \
    import SettingsArgs
from anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time
"""



input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class DataTransform:
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Data transformator: settings was set.")
        self._print_logs(f"{get_current_time()} Data transformator: Visualisate = {self.args.visualisate}")
        self._print_logs(f"{get_current_time()} Data transformator: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Data transformator: Data read!")
        self.input_dict = dictionary
        self.raw_data = self.input_dict["data_body"]["raw_data"]
        self.raw_lables = self.input_dict["data_body"]["raw_columns"]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Data transformator: Start transforming...")
        self._all_data_transorm()
        self._print_logs(f"{get_current_time()} Data transformator: Transforming finished!")

    def output_data(self) -> dict:
        self.input_dict["data_body"]["transformed_data"] = self.transformed_data
        return self.input_dict

    def _all_data_transorm(self) -> None:
        for data in self.raw_data:
            self.transformed_data.append(self._data_transform(data))

    def _data_transform(self, data) -> list:
        self.temp_transformed_data = []
        # Creating list of lists of suitable shape
        for _ in range(len(data[0])):
            self.temp_transformed_data.append([])
        for i, line in enumerate(data):
            for j, element in enumerate(line):
                self.temp_transformed_data[j].append(element)
        return self.temp_transformed_data

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
