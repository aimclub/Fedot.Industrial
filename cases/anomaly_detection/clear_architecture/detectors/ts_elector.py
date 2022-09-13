from cases.anomaly_detection.clear_architecture.detectors.AbstractDetector import AbstractDetector
from cases.anomaly_detection.clear_architecture.utils.get_time import time_now


class TsElector(AbstractDetector):
    """
    Time series elector.
        input format: dict with "data" and "labels" fields
        Output: the same dict but with additional list of window
    """
    def __init__(self, ts_labels):
        super().__init__(name='Time Series Elector', operation='election')
        self.columns_labels = None
        self.transformed_data = None
        self.ts_labels = ts_labels

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{time_now()} {self.name}: Data read!")
        self.input_dict = dictionary
        self.transformed_data = self.input_dict["data_body"]["transformed_data"]
        self.columns_labels = self.input_dict["data_body"]["raw_columns"]

    def output_data(self) -> dict:
        self.input_dict["data_body"]["elected_data"] = self.new_data
        self.input_dict["data_body"]["labels_of_elected_ts"] = self.ts_labels
        return self.input_dict

    def _do_analysis(self) -> None:
        self.new_data = []
        for dataset in self.transformed_data:
            self.new_data.append(self._elect_ts_from_data(dataset))

    def _elect_ts_from_data(self, ts: list) -> list:
        temp_data_set = []
        for column in self.ts_labels:
            data = ts[self.columns_labels.index(column)]
            temp_data_set.append(data)
        return temp_data_set
