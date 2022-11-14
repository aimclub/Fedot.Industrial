from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, \
    ELECTED_DATA, LIST_OF_WINDOWS, \
        STEP_OF_WINDOWS, LEN_OF_WINDOWS
"""
╔═ DATA_TYPE: flag of status of pypeline, currently unused
    ╠═ DATA_BODY: {
    ║   ╠═ RAW_LABLES: lables in format: [x, x, .., x] where x 0 or 1
    ║   ╠═ LABLES_FOR_VISUALISATION: lables in format: [x, x, .., x] where x 0 or n, n could be any
    ║   ╠═ RAW_DATA: data from files, in horizonal format: [[field1_1, field2_1, .. , fieldN_1], [field1_2, field_2_2, .. , fieldN_2], .., []]
    ║   ╠═ COLUMNS_LABLES: columns names
    ║   ╠═ TRANSFORMED_DATA: data in vertical format : [[field1_1, field1_2, .. , field1_N], [field2_1, field2_2, .. , field2_N], .., []]
    ║   ╠═ ELECTED_LABLES: from transformer data we have to elect some data by this lables
    ║   ╠═ ELECTED_DATA: data, choosed by ELECTED_LABLES
    ║   ║ --------------------------    
    ║   ║ ++ We are here: ++
    ║   ╠═ LIST_OF_WINDOWS: list of windows in which we cut data by STEP_OF_WINDOWS and LEN_OF_WINDOWS
    ║   ╠═ STEP_OF_WINDOWS: step of windows
    ║   ╠═ LEN_OF_WINDOWS: length of windows
    ║   ║ --------------------------
    ║   ╚═ DETECTIONS
    ║       ╠═ RAW_PREDICTIONS: raw predictions from detector, stacked!
    ║       ╠═ QUANTILE_PREDICTIONS: predictions from each detector filtered by respected quantile
    ║       ╠═ PREDICTIONS_FOR_VISUALIZATION
    ║       ╠═ ENSAMBLED_PREDICTION: prediction made from all predictions
    ║       ╠═ MAIN_METRIC: main metric of ensambling
    ╚═ DATA_FLAGS: currently unused


input format:

    dict with "data" and "lables" fields

Output 
    the same dict but with additional with windiows_list and lables for it?
    
"""
class WindowCutter(AbstractDataOperation):
    """
    Simple element that cut data in windows of constant len

    Returns:
        _type_: _description_
    """
    args: SettingsArgs

    def __init__(self, window_len, window_step):
        self.window_len = window_len
        self.window_step = window_step
        self.output_window_list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Window cutter: settings was set.")
        self._print_logs(f"{get_current_time()} Window cutter: Visualisate = {self.args.visualisate}")
        self._print_logs(f"{get_current_time()} Window cutter: Print logs = {self.args.print_logs}")

    def input_data(self, input_dict: dict) -> None:
        self._print_logs(f"{get_current_time()} Window cutter: Data read!")
        self.input_dict = input_dict
        self.data = self.input_dict[DATA_BODY][ELECTED_DATA]

    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Window cutter: Start cutting...")
        self._cut_data_to_windows()
        self._print_logs(f"{get_current_time()} Window cutter: Cutting finished!")


    def output_data(self) -> dict:
        self.input_dict[DATA_BODY][LIST_OF_WINDOWS] = self.output_window_list
        self.input_dict[DATA_BODY][LEN_OF_WINDOWS] = self.window_len
        self.input_dict[DATA_BODY][STEP_OF_WINDOWS] = self.window_step
        return self.input_dict

    def _cut_data_to_windows(self) -> None:
        for data in self.data:
            windows = self._cut_ts_to_windows(data)
            self.output_window_list.append(windows)

    def _cut_ts_to_windows(self, ts: list) -> list:
        start_idx = 0
        end_idx = len(ts[0]) - self.window_len
        temp_windows_list = []
        for i in range(start_idx, end_idx, self.window_step):
            temp_window = []
            for _ in ts:
                temp_window.append([])
            for j in range(i, i + self.window_len):
                for t in range(len(temp_window)):
                    temp_window[t].append(ts[t][j])
            temp_windows_list.append(temp_window)

        return temp_windows_list


        
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)
