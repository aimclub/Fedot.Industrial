from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time

from sklearn.metrics import f1_score

from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, LABLES_FOR_VISUALIZATION, \
        PREDICTS_ZONES, ENSAMBLED_PREDICTION
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
║   ╠═ LIST_OF_WINDOWS: list of windows in which we cut data by STEP_OF_WINDOWS and LEN_OF_WINDOWS
║   ╠═ STEP_OF_WINDOWS: step of windows
║   ╠═ LEN_OF_WINDOWS: length of windows
║   ╚═ DETECTIONS
║       ╠═ RAW_PREDICTIONS: raw predictions from detector, stacked!
║       ╠═ QUANTILE_PREDICTIONS: predictions from each detector filtered by respected quantile
║       ╠═ PREDICTIONS_FOR_VISUALIZATION
║       ╠═ ENSAMBLED_PREDICTION: prediction made from all predictions
║       ╠═ MAIN_METRIC: main metric of ensambling
║       ║ --------------------------    
║       ║ ++ We are here: ++
║       ║ PREDICTS_ZONES: zones of predicted anomalies to get metrics for recognition
║       ║ --------------------------
╚═ DATA_FLAGS: currently unused
    
"""
class PredictsZones(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: settings was set.")
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Data read!")
        self.input_dict = dictionary
        #self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        #self.lables = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        #self.lables_for_metrics = self.input_dict[DATA_BODY][RAW_LABLES]
        self.ensambled_predictions = self.input_dict[DATA_BODY][DETECTIONS][ENSAMBLED_PREDICTION]


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Loading metrics...")
        self._Metrics()
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Ready!")

    def output_data(self) -> dict:
        self.input_dict[DATA_BODY][DETECTIONS][PREDICTS_ZONES] = self.list_preds
        return self.input_dict

    def _Metrics(self) -> None:
        self.list_preds = []
        for prediction in self.ensambled_predictions:
            temp_anomalies = []
            state = 0 # 1 in, 2 out
            start = 0
            end = 0
            for i, item in enumerate(prediction):
                if state == 0:
                    if item == 0:
                        state = 2
                    else:
                        state = 1
                        start = i
                    continue
                if state == 1:
                    if item != 1:
                        end = i
                        temp_anomalies.append([start, end])
                        start = i
                        state = 2
                    continue
                if state == 2:
                    if item == 1:
                        state = 1
                        start = i
                    continue
            if start != end:
                temp_anomalies.append([start, len(prediction)-1])
            #print(temp_anomalies)
            self.list_preds.append(temp_anomalies)         
    
    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)

