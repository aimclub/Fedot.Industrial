from enum import Enum

from fedot_ind.core.architecture.experiment.computer_vision import CVExperimenter
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierNN import TimeSeriesClassifierNN
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.architecture.experiment.TimeSeriesRegression import TimeSeriesRegression
from fedot_ind.core.ensemble.static.RankEnsembler import RankEnsemble


class EnsembleEnum(Enum):
    Rank_Ensemble = RankEnsemble


class TaskEnum(Enum):

    ts_classification = dict(fedot_preset=TimeSeriesClassifierPreset,
                             nn=TimeSeriesClassifierNN,
                             default=TimeSeriesClassifier)

    ts_regression = (TimeSeriesRegression,)
    anomaly_detection = (TimeSeriesClassifier,)
    image_classification = (CVExperimenter,)
    object_detection = (CVExperimenter,)
    semantic_segmentation = (CVExperimenter,)
