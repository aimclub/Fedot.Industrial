

from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation

from cases.anomaly_detection.utils.get_time \
    import get_current_time
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.constants.data_types \
    import CLEAR_DATA
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, DATA_TYPE, RAW_DATA, \
    LABLES_FOR_VISUALIZATION, COLUMNS_LABLES, \
    RAW_LABLES, DATA_FLAGS
"""
╔═ DATA_TYPE: flag of status of pypeline, currently unused
    ╠═ DATA_BODY: {
    ║   ║ --------------------------    
    ║   ║ ++ We are here: ++
    ║   ╠═ RAW_LABLES: lables in format: [x, x, .., x] where x 0 or 1
    ║   ╠═ LABLES_FOR_VISUALISATION: lables in format: [x, x, .., x] where x 0 or n, n could be any
    ║   ╠═ RAW_DATA: data from files, in horizonal format: [[field1_1, field2_1, .. , fieldN_1], [field1_2, field_2_2, .. , fieldN_2], .., []]
    ║   ╠═ COLUMNS_LABLES: columns names
    ║   ║ --------------------------
    ║   ╠═ TRANSFORMED_DATA: data in vertical format : [[field1_1, field1_2, .. , field1_N], [field2_1, field2_2, .. , field2_N], .., []]
    ║   ╠═ ELECTED_LABLES: from transformer data we have to elect some data by this lables
    ║   ╠═ ELECTED_DATA: data, choosed by ELECTED_LABLES
    ║   ╠═ LIST_OF_WINDOWS: list of windows in which we cut data by STEP_OF_WINDOWS and LEN_OF_WINDOWS
    ║   ╠═ STEP_OF_WINDOWS: step of windows
    ║   ╠═ LEN_OF_WINDOWS: length of windows
    ║   ╚═ DETECTIONS
    ║       ╠═ RAW_PREDICTIONS: raw predictions from detector, stacked!
    ║       ╠═ QUANTILE_PREDICTIONS: predictions from each detector filtered by respected quantile
    ║       ╠═ PREDICTIONS_FOR_VISUALIZATION
    ║       ╠═ ENSAMBLED_PREDICTION: prediction made from all predictions
    ║       ╠═ MAIN_METRIC: main metric of ensambling
    ╚═ DATA_FLAGS: currently unused
"""
class DataCleaner(AbstractDataOperation):
    """
    This element of pipline make one single thing - just clear pipline data and return it to 
    next-after-read-data state

    """
    args: SettingsArgs

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Cleaner settings was set.")
        self._print_logs(f"{get_current_time()} Cleaner Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Cleaner Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self.input_dict = dictionary
        self.raw_data = self.input_dict[DATA_BODY][RAW_DATA]
        self.raw_lables = self.input_dict[DATA_BODY][COLUMNS_LABLES]
        self.raw_lables_1 = self.input_dict[DATA_BODY][RAW_LABLES]
        self.raw_lables_for_show = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        self._print_logs(f"{get_current_time()} Cleaner Start reading...")


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Cleaner Data is ready!")

    def output_data(self) -> dict:
        #self.all_lables.extend(["Tr1", "Tr2"])
        return {
            DATA_TYPE: CLEAR_DATA,
            DATA_BODY: {
                RAW_LABLES: self.raw_lables_1,
                RAW_DATA: self.raw_data, 
                COLUMNS_LABLES: self.raw_lables,
                LABLES_FOR_VISUALIZATION: self.raw_lables_for_show
                },
            DATA_FLAGS: {}
        }

    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)


