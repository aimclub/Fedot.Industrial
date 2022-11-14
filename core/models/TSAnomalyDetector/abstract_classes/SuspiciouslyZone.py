
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


class SuspiciouslyZone:
    def __init__(self, start: int, end: int, metric: float) -> None:
        self.start: int = start
        self.end: int = end
        self.metric: float = metric
    
    def get_start(self) -> int:
        return self.start
        
    def get_end(self) -> int:
        return self.end

    def get_metric(self) -> float:
        return self.metric