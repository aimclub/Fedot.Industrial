from typing import List, Mapping


class AnomalyZone:
    """
        + This class contains all data of one anomaly +
        Be careful, this class uses both for dataset objects of anomalies and for 
        anomalies that we are looking at!
    """
    def __init__(self, start: int = None, end: int = None) -> None:
        # Common fields
        # start and end counts of anomaly on data
        self.start: int = start
        self.end: int = end
        # parts of data that contains in this anomaly in form of 
        # an Mapping[str, list]
        self.data: Mapping[str, list] = {}
        # Additional time series that was generated for anomaly
        # Max and min time series that formed by choosing max and min of all 
        # data time seris en every point of anomaly
        self.min_data:  List[float] = None
        self.max_data:  List[float] = None
        # distance data - time seres that creates from distances from max and min
        # additional ts's in each point of anomaly
        self.distance_data: List[float] = None
        # ts like upper one, but cutted by threshold in Simple Detector element
        self.distance_data_trans: List[float] = None
        # additional ts that are max of average absolute deviation of all ts's in main data
        self.average_absolute_deviation: List[float] = None
        self.average_absolute_deviation_transformed: List[float] = None
        self.additional_mean_ts: List[float] = None
        # vector of features for searching anomalie's heaviness
        self.features_for_heaviness_clusterization: List[float] = []
        # vector of features for comparion anomalie to dataset of anomalies
        self.features_for_datasets: List[float] = []
        # vector of features for searching anomalie's type
        self.features_for_type_clusterization: List[float] = []

        # Type, description and heaviness of anomaly
        self.type: str = None
        self.comment: str = "None"
        self.heaviness: int = 0

        # For adataset usage - type of anomaly
        self.anomaly_class: int = None
        # For adataset usage - names of features
        self.features: List[str] = []
        # ???
        self.data_for_dataset: dict  = {}
        # +++ Threshold for <self.distance_data_trans> field +++
        self.threshold = 0
        # Fields for visualisation 
        # still in progress...
        self.x: float = 0
        self.y: float = 0
        # coordinates for visualisation of clusters 
        # during search of heaviness of animaly
        self.x_heaviness: float = 0
        self.y_heaviness: float = 0
        # coordinates for visualisation of clusters 
        # during search of type of animaly
        self.x_type: float = 0
        self.y_type: float = 0

        # predicted information
        # type and heaviness of anomaly that was get by comparing this anomaly
        # to dataset of anomalies 
        self.dataset_type: int = 0
        self.dataset_heaviness: int = 0
        self.dataset_comment: str = ""
        # type and heaviness of anomaly that was get by clusterizations
        self.cluster_type: int = 0
        self.cluster_heaviness: int = 0
        self.cluster_comment: str = ""
        # resulted type and heaviness of anomaly 
        self.predicted_type: int = 0
        self.predicted_heaviness: int = 0
        self.predicted_comment: str = ""


    def get_start(self) -> int:
        return self.start
        
    def get_end(self) -> int:
        return self.end
    
    
