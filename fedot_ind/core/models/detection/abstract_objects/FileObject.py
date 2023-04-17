from typing import List, Mapping

from fedot_ind.core.models.detection.abstract_objects.AnomalyZone import AnomalyZone


class FileObject:
    """
    This class contains all data that contains in one file - time series, lables,
    transformation params, anomalies list
    """
    def __init__(self, 
            test_data: List[float], 
            filename: str
        ) -> None:
        # test data
        self.test_data = test_data
        # all data from file in form of dictioary
        self.time_series_data: Mapping[str, List[float]] = {}
        # list of nomalies in this data
        self.anomalies_list: List[AnomalyZone] = []
        # length of data in file
        self.data_length: int = None
        # filename and full path to file
        self.filename: str = filename
        #self.filepath: str = filepath
        # lables list(or dict???)
        self.lables: list = None
        self.elected_columns: List[str] = None
        self.current_elected_columns: List[str] = []

        # main transformed data
        self.transformed_data: Mapping[str, List[float]] = {}

        # additional ts that creates durnig analysis
        self.additional_min_ts: List[float] = None
        self.additional_max_ts: List[float] = None
        self.additional_distance_ts: List[float] = None
        self.additional_average_absolute_deviation_max: List[float] = None
        self.additional_average_absolute_deviation_sum: List[float] = None
        self.additional_transformed_average_absolute_deviation: List[float] = None
        self.additional_mean_ts: List[float] = None
        # threshold for <additional_transformed_average_absolute_deviation>
        self.threshold: float = None
        self.test_vector_ts = []

    def get_len_of_dataset(self) -> int:
        """
        This method need to make things more standart. 
        Returns len of data in this FileObject

        Returns:
            int: len of data
        """
        return len(self.test_data)
    
    def get_lables_list(self, min_heavines: int = 0) -> list:
        if len(self.lables):
            temp_list = []
            for predict in self.lables:
                if predict[2] > min_heavines:
                    temp_list.append(predict)
            return temp_list
        else:
            return False
    
    def get_lables_for_metrics(self, min_heavines: int = 0) -> List[int]:
        """
        Returns lables in suitable format for metrics: [0, 0, 0, 1, 1, 1, ..., 0]

        Args:
            min_heavines (int, optional): Set this arg to none-zero values if you 
            need to filter lables by heaviness of anomalies. Defaults to 0.

        Returns:
            List[int]: list in format [0, 0, 0, 1, 1, 1, ..., 0] with len the same 
            as len of data in this object
        """
        output_list = [0] * self.get_len_of_dataset()
        if len(self.lables):
            for predict in self.lables:
                if predict[2] > min_heavines:
                    begin = predict[0]
                    end = predict[1]
                    if begin < 0: begin = 0
                    if end >= self.get_len_of_dataset(): end = self.get_len_of_dataset() - 1 
                    for i in range(begin, end):
                        output_list[i] = 1
            return output_list
        else:
            return output_list
    
    def get_all_predicts_list_by_type(self, type_numbers: List[int]) -> List[AnomalyZone]:
        """
        This method returns list of predicts by their class
        or all predicts if type_numbers set to -1
        if type_number == -1 - returns all predicts
        Args:
            type_number (List[int]): could contain any of this numbers [-1, 0, 1, 2, 3]

        Returns:
            List[AnomalyZone]: list of anomalies
        """
        if not len(self.anomalies_list):
            return False
        else: 
            if -1 in type_numbers:
                predictions_list = []
                for predict in self.anomalies_list:
                    predictions_list.append(predict)
            else:
                predictions_list = []
                for predict in self.anomalies_list:
                    if predict.predicted_type in type_numbers: # and predict.heaviness >9: 
                    #if predict.cluster_type == type_number: dataset_type predicted_type
                        predictions_list.append(predict)
            return predictions_list


    def get_predict_for_metrics(self, classes: List[int] =[2, 3]) -> List[int]:
        """
        Returns predicts in suitable format for metrics: [0, 0, 0, 1, 1, 1, ..., 0]
        arg <classes> sets classes of predicts that will be included into output

        Args:
            classes (List[int], optional): could be any of [0, 1, 2, 3]. Defaults to [2, 3].

        Returns:
            List[int]: list in format [0, 0, 0, 1, 1, 1, ..., 0] with len the same 
            as len of data in this object
        """
        time_series_of_predict_for_metrics = [0] * self.get_len_of_dataset()

        if len(self.anomalies_list) == 0:
            return time_series_of_predict_for_metrics
        else:
            for zone in self.anomalies_list:
                if zone.predicted_type in classes:
                    for j in range(zone.get_start(), zone.get_end(), 1):
                        if 0 <= j < self.get_len_of_dataset():
                            time_series_of_predict_for_metrics[j] = 1
        return time_series_of_predict_for_metrics

    def copy_parts_of_datasets_in_anomaly_zones(self) -> None:
        """
        This method copy parts of raw data to anomaly zones
        """
        for i in range(len(self.anomalies_list)):
            if len(self.time_series_data.keys()):
                for column in list(self.time_series_data.keys()):
                    # Copy data of each time series
                    self.anomalies_list[i].data[column] = \
                        self.time_series_data[column][
                            self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                            ]
            # Copy additional_min_ts
            if self.additional_min_ts  is not None:
                self.anomalies_list[i].min_data = self.additional_min_ts[
                    self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                ]
            # Copy additional_max_ts
            if self.additional_max_ts is not None:
                self.anomalies_list[i].max_data = self.additional_max_ts[
                    self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                ]
            # Copy distance_data
            if self.additional_distance_ts is not None:
                self.anomalies_list[i].distance_data = \
                    self.additional_distance_ts[
                            self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                        ]
            if self.additional_average_absolute_deviation_max is not None:
            # Copy average_absolute_deviation
                self.anomalies_list[i].average_absolute_deviation = \
                    self.additional_average_absolute_deviation_max[
                            self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                        ]
            # Copy average_absolute_deviation_transformed
            if self.additional_transformed_average_absolute_deviation is not None:
                self.anomalies_list[i].average_absolute_deviation_transformed = \
                    self.additional_transformed_average_absolute_deviation[
                            self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                        ]

            # Copy additional_mean_ts
            if self.additional_mean_ts is not None:
                self.anomalies_list[i].additional_mean_ts = \
                    self.additional_mean_ts[
                            self.anomalies_list[i].get_start():self.anomalies_list[i].get_end()
                        ]
            # Copy threshold
            self.anomalies_list[i].threshold = self.threshold
