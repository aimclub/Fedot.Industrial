
"""
    Description:

    Class contains data of one anomaly zone after detection of zones. 
    Data contains:

        start-end of zone
        time series' number
        cluster number
        type of anomaly
        description of the type?

    Methods
        get_features
        update ts_data?





"""


class AnomalyZone:
    def __init__(self, start: int = None, end: int = None) -> None:
        self.start: int = start
        self.end: int = end
        self.data: dict = {}
        self.data_for_dataset: dict  = {}
        self.type: str = None
        self.features: list = []
        self.features_for_datasets: list = []
        self.comment: str = "None"
        self.heaviness: int = 0
        self.anomaly_class: int = None
        self.distance_data_trans: list = []
        self.min_data: list = None
        self.max_data: list = None
        self.distance_data: list = None

        self.x: float = 0
        self.y: float = 0
        self.cluster_type: int = None
        self.dataset_type: int = None
        self.predicted_type: int = 0
        
    def get_start(self) -> int:
        return self.start
        
    def get_end(self) -> int:
        return self.end
    


    
