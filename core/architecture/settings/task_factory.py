from enum import Enum
from core.ensemble.static.RankEnsembler import RankEnsemble
from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesClassifierNN, \
    TimeSeriesImageClassifier


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):
    ts_classification = (TimeSeriesClassifier, TimeSeriesClassifierNN)
    image_classification = (TimeSeriesImageClassifier)
    anomaly_detection = (TimeSeriesClassifier)
    object_detection = (TimeSeriesClassifier)
