import numpy as np

from core.models.detection.utils.get_time import time_now
from core.models.detection.abstract_objects.FileObject import FileObject


class AbstractDataOperation:

    def set_settings(self):
        self.name = ""
        self.print_logs: bool = True
        self._print_logs("Settings was set.")

    def load_data(self, object) -> None:
        raise NotImplementedError()

    def run_operation(self) -> None:
        self._print_logs(f"Start {self.operation}...")
        self._print_logs(f"{self.operation} finished!")

    def return_new_data(self):
        raise NotImplementedError()

    def _print_logs(self, log_message: str) -> None:
        if self.print_logs:
            print(f"---[{time_now()}] {self.name}: {log_message}")


