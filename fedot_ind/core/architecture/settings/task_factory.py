from enum import Enum

from fedot_ind.core.architecture.experiment.TimeSeriesAnomalyDetection import TimeSeriesAnomalyDetectionPreset
from fedot_ind.core.architecture.experiment.computer_vision import CVExperimenter
from fedot_ind.core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierNN import TimeSeriesClassifierNN
from fedot_ind.core.architecture.experiment.TimeSeriesClassifierPreset import TimeSeriesClassifierPreset
from fedot_ind.core.architecture.experiment.TimeSeriesRegression import TimeSeriesRegression
# from fedot_ind.core.architecture.experiment import TimeSeriesForecasingWithDecomposition
from fedot_ind.core.ensemble.rank_ensembler import RankEnsemble


class EnsembleEnum(Enum):
    Rank_Ensemble = RankEnsemble


class TaskEnum(Enum):

    ts_classification = dict(fedot_preset=TimeSeriesClassifierPreset,
                             nn=TimeSeriesClassifierNN,
                             default=TimeSeriesClassifier)
    # ts_forecasting = (TimeSeriesForecasingWithDecomposition,)
    ts_regression = (TimeSeriesRegression,)
    anomaly_detection = (TimeSeriesAnomalyDetectionPreset,)
    image_classification = (CVExperimenter,)
    object_detection = (CVExperimenter,)
    semantic_segmentation = (CVExperimenter,)


