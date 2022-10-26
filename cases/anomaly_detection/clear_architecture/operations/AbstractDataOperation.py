import numpy as np

from cases.anomaly_detection.clear_architecture.utils.get_time import time_now
from cases.anomaly_detection.clear_architecture.utils.settings_args import SettingsArgs


class AbstractDataOperation:
    args: SettingsArgs

    def __init__(self,  operation: str, name: str = 'Unknown'):
        self.name = name
        self.operation = operation

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{time_now()} {self.name}: settings was set.")
        self._print_logs(f"{time_now()} {self.name}: Visualize = {self.args.visualize}")
        self._print_logs(f"{time_now()} {self.name}: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        raise NotImplementedError()

    def run(self) -> None:
        self._print_logs(f"{time_now()} {self.name}: Start {self.operation}...")
        self._do_analysis()
        self._print_logs(f"{time_now()} {self.name}: {self.operation} finished!")

    def output_data(self) -> dict:
        raise NotImplementedError()

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)

    def _do_analysis(self) -> None:
        """Abstract method for analysis"""
        raise NotImplementedError()

    @staticmethod
    def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
