from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier, TimeSeriesImageClassifier
from core.ensemble.static.RankEnsembler import RankEnsemble
from core.models.EnsembleRunner import EnsembleRunner
from core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from core.models.signal.SignalExtractor import SignalExtractor
from core.models.spectral.SSAExtractor import SSAExtractor
from core.models.statistical.QuantileRunner import StatsExtractor
from core.models.topological.TopologicalRunner import TopologicalExtractor


class FeatureGenerator(Enum):
    quantile = StatsExtractor
    window_quantile = StatsExtractor
    wavelet = SignalExtractor
    spectral = SSAExtractor
    window_spectral = SSAExtractor
    topological = TopologicalExtractor
    recurrence = RecurrenceExtractor
    ensemble = EnsembleRunner


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):
    ts_classification = TimeSeriesClassifier
    ts_image_classification = TimeSeriesImageClassifier
