from enum import Enum

from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.architecture.experiment.TimeSeriesImageClassifier import TimeSeriesImageClassifier
from core.ensemble.static.RankEnsembler import RankEnsemble
from core.models.EnsembleExtractor import EnsembleExtractor
from core.models.signal.RecurrenceExtractor import RecurrenceExtractor
from core.models.signal.SignalExtractor import SignalExtractor
from core.models.spectral.SSAExtractor import SSAExtractor
from core.models.statistical.StatsExtractor import StatsExtractor
from core.models.topological.TopologicalExtractor import TopologicalExtractor


class FeatureGenerator(Enum):
    quantile = StatsExtractor
    window_quantile = StatsExtractor
    wavelet = SignalExtractor
    spectral = SSAExtractor
    window_spectral = SSAExtractor
    topological = TopologicalExtractor
    recurrence = RecurrenceExtractor
    ensemble = EnsembleExtractor


class EnsembleGenerator(Enum):
    Rank_Ensemble = RankEnsemble


class TaskGenerator(Enum):
    ts_classification = TimeSeriesClassifier
    ts_image_classification = TimeSeriesImageClassifier
