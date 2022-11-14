from cases.anomaly_detection.utils.settings_args \
    import SettingsArgs
from cases.anomaly_detection.utils.get_time \
    import get_current_time

from sklearn.metrics import f1_score

from cases.anomaly_detection.abstract_classes.AbstractDataOperation import AbstractDataOperation
from cases.anomaly_detection.constants.current_data_const import \
    DATA_BODY, RAW_LABLES, RAW_PREDICTIONS,\
    ELECTED_DATA, DETECTIONS, LABLES_FOR_VISUALIZATION, FURCHER_PREDICTS_ZONES,\
        PREDICTS_ZONES, ENSAMBLED_PREDICTION, DETECTIONS_VECTORS, CLUST_NUMBER_FOR_ALL_DATA, FURCHER_DETECTIONS_VECTORS
from cases.anomaly_detection.utils.get_features import generate_features_from_one_ts
from cases.anomaly_detection.utils.math_utils import NormalizeData, NormalizeDataForDetectors
import numpy as np
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
class FurcherPredictsMetricsFromData(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []
    def __init__(self, features: list, cluster_number: int) -> None:
        self.cluster_number = cluster_number
        self.features = features

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: settings was set.")
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Print logs = {self.args.print_logs}")

    def input_data(self, dictionary: dict) -> None:
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Data read!")
        self.input_dict = dictionary
        self.data = self.input_dict[DATA_BODY][ELECTED_DATA]
        #if DETECTIONS_VECTORS not in self.input_dict[DATA_BODY][DETECTIONS]:
        #    self.input_dict[DATA_BODY][DETECTIONS][DETECTIONS_VECTORS] = []
        #self.lables = self.input_dict[DATA_BODY][LABLES_FOR_VISUALIZATION]
        #self.lables_for_metrics = self.input_dict[DATA_BODY][RAW_LABLES]
        self.zones = self.input_dict[DATA_BODY][DETECTIONS][PREDICTS_ZONES]
        self.clusters = self.input_dict[DATA_BODY][DETECTIONS][CLUST_NUMBER_FOR_ALL_DATA]


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Loading metrics...")
        self._Metrics()
        self._print_logs(f"{get_current_time()} Predictions PredictsZones: Ready!")

    def output_data(self) -> dict:
        return self.input_dict

    def _Metrics(self) -> None:
        #self.input_dict[DATA_BODY][DETECTIONS][DETECTIONS_VECTORS]
        temp_vecotrs = []
        temp_zones = []
        for i, data in enumerate(self.data):
            temp_vecotrs.append([])
            temp_zones.append([])
            for j, zone in enumerate(self.zones[i]):
                # format - len, dist, area
                if self.clusters[i][j] == self.cluster_number:
                    temp_zones[-1].append(zone)
                    temp_vecotrs[-1].append([])
                    for ts in data:
                        temp_vecotrs[-1][-1].extend(generate_features_from_one_ts(ts[zone[0]: zone[1]], self.features))
                        #self.input_dict[DATA_BODY][DETECTIONS][DETECTIONS_VECTORS][i][j].extend(generate_features_from_one_ts(ts[zone[0]: zone[1]], self.features))
        temp_data = []
        for dataset in temp_vecotrs:
            for data in dataset:
                temp_data.append(data)
        self.input_dict[DATA_BODY][DETECTIONS][FURCHER_DETECTIONS_VECTORS] = temp_vecotrs
        self.input_dict[DATA_BODY][DETECTIONS][FURCHER_PREDICTS_ZONES] = temp_zones
        #rint(temp_data)
        #path = "/media/nikita/HDD/Data_part_1/metrics/metrics_1.txt"
        #with open(path, 'w') as fp:
        #    for item in temp_data:
        #        # write each item on a new line
        #        fp.write("%s\n" % item)
        print('Done')
    
    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)

