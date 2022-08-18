from cases.anomaly_detection.clear_architecture.utils.get_time \
    import get_current_time
from cases.anomaly_detection.clear_architecture.utils.settings_args \
    import SettingsArgs

"""
input format:
    dict with "data" and "labels" fields

Output 
    the same dict but with additional list of window    
"""


class TsElector:
    args: SettingsArgs

    def __init__(self, ts_labels):
        self.ts_labels = ts_labels

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Time series elector: settings was set.")
        self._print_logs(f"{get_current_time()} Time series elector: Visualize = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Time series elector: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Time series elector: Data read!")
        self.input_dict = dictionary
        self.transformed_data = self.input_dict["data_body"]["transformed_data"]
        self.columns_lables = self.input_dict["data_body"]["raw_columns"]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Time series elector: Start electing...")
        self._elect_data()
        self._print_logs(f"{get_current_time()} Time series elector: Electing finished!")

    def output_data(self) -> dict:
        self.input_dict["data_body"]["elected_data"] = self.new_data
        self.input_dict["data_body"]["labels_of_elected_ts"] = self.ts_labels
        return self.input_dict

    def _elect_data(self) -> list:
        # File level
        self.new_data = []
        for dataset in self.transformed_data:
            self.new_data.append(self._elect_ts_from_data(dataset))

    def _elect_ts_from_data(self, ts: list) -> list:
        temp_data_set = []
        for column in self.ts_labels:
            data = ts[self.columns_lables.index(column)]
            # filtered = lowess(data, is_sorted=True, frac=0.025, it=0)
            temp_data_set.append(data)
        return temp_data_set
        
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
