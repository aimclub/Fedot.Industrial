from TSAnomalyDetector.utils.settings_args \
    import SettingsArgs
from TSAnomalyDetector.utils.get_time \
    import get_current_time
import os
from sklearn.metrics import f1_score
import pickle
from TSAnomalyDetector.abstract_classes.AbstractDataOperation import AbstractDataOperation
from TSAnomalyDetector.abstract_classes.DataObject import DataObject
from TSAnomalyDetector.abstract_classes.AnomalyZone import AnomalyZone

class ClusterizationMetrics(AbstractDataOperation):
    args: SettingsArgs
    raw_data: list
    transformed_data: list = []
    def __init__(self, intersection: float = 0.3, metric: str = "F1") -> None:
        self.intersection = intersection
        self.metric = metric

    def set_settings(self, args: SettingsArgs):
        self.args = args
        self._print_logs(f"{get_current_time()} Clusterization metrics: settings was set.")
        self._print_logs(f"{get_current_time()} Clusterization metrics: Visualisate = {self.args.visualize}")
        self._print_logs(f"{get_current_time()} Clusterization metrics: Print logs = {self.args.print_logs}")

    def input_data(self, data_object: DataObject) -> None:
        self._print_logs(f"{get_current_time()} Clusterization metrics: Data read!")
        self.data_object = data_object


    def run(self) -> None:
        self._print_logs(f"{get_current_time()} Clusterization metrics: Loading metrics...")
        self.metrics_list = []
        for file in self.data_object.get_list_of_files():
            if self.metric != "F1":
                score = self.custom_metric(
                    self.data_object.get_lables_for_metrics(file), 
                    self.data_object.get_predict_for_metrics(file), 
                    self.intersection)
            else:
                score = f1_score(
                    self.data_object.get_lables_for_metrics(file), 
                    self.data_object.get_predict_for_metrics(file), 
                    average='macro')
            self._print_logs(f"{get_current_time()} Metrics: Metric for file {file} - {score}")
            self.metrics_list.append(score)
        self._print_logs(f"{get_current_time()} Clusterization metrics: ----------------------------------------")
        self.main_score = sum(self.metrics_list) / len(self.metrics_list)
        self._print_logs(f"{get_current_time()} Clusterization metrics: Average predict:")
        self._print_logs(f"{get_current_time()} Clusterization metrics: {self.main_score}")
        self._print_logs(f"{get_current_time()} Clusterization metrics: -------------------------------------")
        if False:
            self.path = f"/home/nikita/Desktop/Fedot.Industrial/cases/anomaly_detection/clusterization/{self.main_score}"
            try:
                os.mkdir(self.path)
                path_clust = os.path.join(self.path, "Clusterizator.pickle")
                path_reducer = os.path.join(self.path, "Reducer.pickle")

                with open(path_clust, 'wb') as f:
                    pickle.dump(self.clusterizator, f)
                with open(path_reducer, 'wb') as f:
                    pickle.dump(self.reducer, f)
            except:
                print("Error! File exists!")

        self._print_logs(f"{get_current_time()} Clusterization metrics: Ready!")

    def output_data(self) -> DataObject:
        return self.data_object

    
    def _print_logs(self, log_message: str) -> None:
        if self.args.print_logs:
            print(log_message)


    def custom_metric(self, lables: list, predict: list, percentage: float) -> float:
        if len(lables) != len(predict): raise ValueError("Arrays aren't the same len!")
        counter_of_lables = 0
        counter_of_detections = 0
        current_counter_pred = 0
        current_counter_lables = 0
        for i in range(len(lables)):
            if lables[i] == 1:
                current_counter_lables += 1
                if predict[i] == 1:
                    current_counter_pred += 1
            else:
                if current_counter_lables != 0:
                    counter_of_lables+=1
                    if  current_counter_pred / current_counter_lables>= percentage:
                        counter_of_detections += 1
                    current_counter_pred = 0
                    current_counter_lables = 0
                else:
                    current_counter_pred = 0
                    current_counter_lables = 0
        if lables[-1] == 1:
            counter_of_lables+=1
            if  current_counter_pred / current_counter_lables >= percentage:
                counter_of_detections += 1            
        if counter_of_lables == 0: return 1
        if counter_of_lables > 0:
            return counter_of_detections / counter_of_lables
        return 0