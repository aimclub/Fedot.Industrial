from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.architecture.experiment.TimeSeriesImageClassifier import TimeSeriesImageClassifier
from core.ensemble.static.RankEnsembler import RankEnsemble
from core.models.EnsembleRunner import EnsembleRunner
from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.spectral.SSARunner import SSARunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner


class FeatureGenerator(Enum):
    # WINDOW GEENRATORS
    window_quantile = StatsRunner
    window_spectral = SSARunner

    # NON-WINDOW GENERATORS
    quantile = StatsRunner
    wavelet = SignalRunner
    spectral = SSARunner
    topological = TopologicalRunner
    recurrence = RecurrenceRunner

    # ENSEMBLE GENERATORS
    ensemble = EnsembleRunner


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):
    ts_classification = TimeSeriesClassifier
    ts_image_classification = TimeSeriesImageClassifier
