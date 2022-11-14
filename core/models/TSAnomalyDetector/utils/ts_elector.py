from pandas import array
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation
from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
from statsmodels.nonparametric.smoothers_lowess import lowess
from TSAnomalyDetector.constants.current_data_const import \
    DATA_BODY, COLUMNS_LABLES, TRANSFORMED_DATA, \
    ELECTED_DATA, ELECTED_LABLES
from TSAnomalyDetector.abstract_classes.DataObject import DataObject

"""
input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional list of window
    
"""
class TsElector(AbstractDataOperation):
    """
    Data contains several lines, elector choose some of them for our work

    Returns:
        _type_: _description_
    """
    args: SettingsArgs

    def __init__(self, ts_lables):
        self.ts_lables = ts_lables

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Time series elector: settings was set.")
        self._print_logs(f"{get_current_time()} Time series elector: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Time series elector: Print logs = {self.args.print_logs}")

    def input_data(self, object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Time series elector: Data read!")
        self.input_object = object

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Time series elector: Start electing...")
        self._elect_data()
        self._print_logs(f"{get_current_time()} Time series elector: Electing finished!")

    def output_data(self) -> dict:
        return self.input_object

    def _elect_data(self) -> list:
        # File level
        if self.input_object.check_columns_for_correctness(self.ts_lables):
            self.input_object.current_elected_columns = self.ts_lables
        else:
            raise ValueError("Elected olumns are not common for every dataset!")
  
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
