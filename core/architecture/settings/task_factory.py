from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesClassifierNN, \
    TimeSeriesImageClassifier
from core.architecture.experiment.TImeSeriesClassifierPreset import TimeSeriesClassifierPreset
from core.ensemble.static.RankEnsembler import RankEnsemble


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):

    ts_classification = dict(fedot_preset=TimeSeriesClassifierPreset,
                             nn=TimeSeriesClassifierNN,
                             default=TimeSeriesClassifier)

    image_classification = (TimeSeriesImageClassifier,)
    anomaly_detection = (TimeSeriesClassifier,)
    object_detection = (TimeSeriesClassifier,)
