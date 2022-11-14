import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(os.path.join(import_path, "../.."))
from models.ExperimentRunner import ExperimentRunner
from TSAnomalyDetector.MainPipeline import MainPipeline

# Reader of time series and lables
from TSAnomalyDetector.utils.read_data import DataReader
# Reader of database
from TSAnomalyDetector.utils.dataset_read import DatasetReader
# Data transformer
from TSAnomalyDetector.utils.transform_data import DataTransformer
# Elector of time series
from TSAnomalyDetector.utils.ts_elector import TsElector
# Time series visualizator
from TSAnomalyDetector.visualisation.visualisation import DataVisualizatorNew
# Detector of suspition zones
from TSAnomalyDetector.good_detectors.simple_detector import SimpleDetector
# Clusterization module
from TSAnomalyDetector.clusterization.clusterization import Clusterization
# Features extracotr
from TSAnomalyDetector.metrics.predicts_metrics_from_data import GetPredictsFeaturesFromData
# Metric extractor
from TSAnomalyDetector.metrics.clusterization_metrics import ClusterizationMetrics

class TSAnomalyDetectorRunner(ExperimentRunner):
    """Class for extracting topological features from time series data.

    Args:
        use_cache: flag for using cache

    """

    def __init__(
            self, 
            csv_files_path: str, 
            lables_files_path: str, 
            use_cache: bool = False
            ):
        super().__init__()
        self.use_cache = use_cache
        self.csv_files_path: str = csv_files_path
        self.lables_files_path: str = lables_files_path
        self.pipe = MainPipeline()
        dataset_path = '/home/nikita/Documents/Fedot.Industrial/core/models/TSAnomalyDetector/database/clear_3/'

        self.data_reader = DataReader(self.csv_files_path, self.lables_files_path)
        self.dataset_reader = DatasetReader(dataset_path)

        self.pipe.init_pipeline = \
            [
                self.data_reader,
                self.dataset_reader
            ]
        self.pipe.init_run()
        self.transformer = DataTransformer(4, 0.55)
        labels = ["Vd", "Vu"]
        self.elector = TsElector(labels)

        self.simple_detector = SimpleDetector(0.58)

        self.visualizer = DataVisualizatorNew()
        self.clust = Clusterization(1, 2)

    def get_features(
            self, 
            features_for_clustering: list = [
                #"mean", 
                #"median",
                #"lambda_less_zero", 
                "std", 
                "var", 
                #"max",
                #"min", 
                #"q5", 
                #"q25", 
                #"q75", 
                #"q95", 
                "sum",
                "len",
                "delta",
                #"dist",
                "v10",
                #"v40",
                #"v60",
                #"v80",
            ], 
            features_for_dataset_analysis: str = None
        ):
        if features_for_dataset_analysis == None: features_for_dataset_analysis = features_for_clustering
        self.pred_features_data = GetPredictsFeaturesFromData(
                features_for_clustering, features_for_dataset_analysis
            )
        self.clust_metrics = ClusterizationMetrics(0.2, "Custom")
        self.pipe.detector_pipeline = \
        [
            self.elector,
            self.transformer,
            self.simple_detector,
            self.pred_features_data,
            self.clust,
            self.clust_metrics,
            self.visualizer
        ]
        self.pipe.detector_run()
        return 

    
