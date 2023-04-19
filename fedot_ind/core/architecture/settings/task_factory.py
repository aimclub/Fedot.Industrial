from enum import Enum

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesClassifierNN, \
    TimeSeriesImageClassifier
from fedot_ind.core.architecture.experiment.TImeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.ensemble.static.RankEnsembler import RankEnsemble


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):

    ts_classification = dict(fedot_preset=TimeSeriesClassifierPreset,
                             nn=TimeSeriesClassifierNN,
                             default=TimeSeriesClassifier)

    image_classification = (TimeSeriesImageClassifier,)
    anomaly_detection = (TimeSeriesClassifier,)
    object_detection = (TimeSeriesClassifier,)
