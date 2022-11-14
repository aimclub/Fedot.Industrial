from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
from tqdm import tqdm

from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation

from TSAnomalyDetector.utils.get_features import generate_features_from_one_ts
from TSAnomalyDetector.abstract_classes.DataObject import DataObject

class GetPredictsFeaturesFromData(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []
    def __init__(self, features: list, features_for_datasets_comparing: list) -> None:
        self.features = features
        self.features_for_datasets_comparing = features_for_datasets_comparing

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Metrics from prediction: settings was set.")
        self._print_logs(f"{get_current_time()} Metrics from prediction: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Metrics from prediction: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Metrics from prediction: Data read!")
        self.data_object = data_object


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Metrics from prediction: Loading metrics...")
        self._Metrics()
        self._print_logs(f"{get_current_time()} Metrics from prediction: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

    def _Metrics(self) -> None:
        # Get the same metrics for database and for zones
        if False:
            for i in tqdm(range(len(self.data_object.database))):
                if bool(self.data_object.database[i].data):
                    self.data_object.database[i].features = []
                    for key in self.data_object.current_elected_columns:
                        self.data_object.database[i].features.extend(
                            generate_features_from_one_ts(self.data_object.database[i].data[key], 
                            self.features_for_datasets_comparing))
        
        
        for filename in self.data_object.get_list_of_files():
            if len(self.data_object.ensambled_prediction[filename]) == 0:
                self._print_logs(f"{get_current_time()} Metrics from prediction: WARNING! For file {filename} no anomalies found!")
            else:
                for i in tqdm(range(len(self.data_object.ensambled_prediction[filename]))):
                    self.data_object.ensambled_prediction[filename][i].features = []
                    self.data_object.ensambled_prediction[filename][i].features_for_datasets  = []
                    #for key in self.data_object.current_elected_columns:
                    #    self.data_object.ensambled_prediction[filename][i].features.extend(
                    #            generate_features_from_one_ts(
                    #                self.data_object.ensambled_prediction[filename][i].data[key],
                    #                self.features)
                    #        )
                    self.data_object.ensambled_prediction[filename][i].features.extend(
                                generate_features_from_one_ts(
                                self.data_object.ensambled_prediction[filename][i].distance_data,
                                self.features)
                    )
                    keys = ["Xu", "Yu", "Zu", "Vu"]
                    #for key in keys:
                    #    self.data_object.ensambled_prediction[filename][i].features_for_datasets.extend(
                    #            generate_features_from_one_ts(
                    #                self.data_object.ensambled_prediction[filename][i].data_for_dataset[key],
                    #                self.features_for_datasets_comparing)
                    #        ) #zone.distance_data
                    self.data_object.ensambled_prediction[filename][i].features_for_datasets.extend(
                                generate_features_from_one_ts(
                                self.data_object.ensambled_prediction[filename][i].distance_data,
                                self.features_for_datasets_comparing)
                    )
    
    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)

