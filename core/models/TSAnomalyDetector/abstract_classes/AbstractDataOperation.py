import numpy as np

from TSAnomalyDetector.utils.get_time import time_now
from TSAnomalyDetector.utils.settings_args import SettingsArgs
from TSAnomalyDetector.abstract_classes.DataObject import DataObject


class AbstractDataOperation:
    args: SettingsArgs

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{time_now()} {self.name}: settings was set.")
        self._print_logs(f"{time_now()} {self.name}: Visualize = {self.args.visualize}")
        self._print_logs(f"{time_now()} {self.name}: Print logs = {self.args.print_logs}")

    def input_data(self, object: DataObject) -> None:
        raise NotImplementedError()

    def run(self) -> None:
        self._print_logs(f"{time_now()} {self.name}: Start {self.operation}...")
        self._print_logs(f"{time_now()} {self.name}: {self.operation} finished!")

    def output_data(self) -> DataObject:
        raise NotImplementedError()

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)


