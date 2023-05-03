from enum import Enum

from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesClassifierNN, \
    TimeSeriesImageClassifier
from fedot_ind.core.architecture.experiment.TImeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.architecture.experiment.computer_vision import CVExperimenter
from fedot_ind.core.ensemble.static.RankEnsembler import RankEnsemble


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):

    ts_classification = dict(fedot_preset=TimeSeriesClassifierPreset,
                             nn=TimeSeriesClassifierNN,
                             default=TimeSeriesClassifier)

    anomaly_detection = (TimeSeriesClassifier,)
    image_classification = (CVExperimenter,)
    object_detection = (CVExperimenter,)
    semantic_segmentation = (CVExperimenter,)
