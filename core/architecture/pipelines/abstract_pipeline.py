import pandas as pd
from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.models.signal.RecurrenceRunner import RecurrenceRunner
from core.models.signal.SignalRunner import SignalRunner
from core.models.statistical.QuantileRunner import StatsRunner
from core.models.topological.TopologicalRunner import TopologicalRunner
from core.operation.transformation.basis.chebyshev import ChebyshevBasis
from core.operation.transformation.basis.data_driven import DataDrivenBasis
from core.operation.transformation.basis.fourier import FourierBasis
from core.operation.transformation.basis.legendre import LegenderBasis
from core.operation.transformation.basis.power import PowerBasis


class AbstractPipelines:
    def __init__(self, train_data, test_data):
        self.train_features = train_data[0]
        self.train_target = train_data[1]
        self.test_features = test_data[0]
        self.test_target = test_data[1]
        self.basis = None

        self.basis_dict = {'Legender': LegenderBasis,
                           "Chebyshev": ChebyshevBasis,
                           'DataDriven': DataDrivenBasis,
                           'Power': PowerBasis,
                           'Fourier': FourierBasis}

        self.feature_generator_dict = {'Statistical': StatsRunner,
                                       'Wavelet': SignalRunner,
                                       'Topological': TopologicalRunner,
                                       'Reccurence': RecurrenceRunner
                                       }

        self.generators_with_matrix_input = ['Topological',
                                             'Wavelet',
                                             'Reccurence',
                                             'Statistical']

    def _evaluate(self, classificator, train_features, test_features):
        fitted_model = classificator.fit(train_features=train_features,
                                         train_target=self.train_target)
        predicted_probs_labels = (classificator.predict(test_features=test_features),
                                  classificator.predict_proba(test_features=test_features))
        metrics = PerformanceAnalyzer().calculate_metrics(target=self.test_target,
                                                          predicted_labels=predicted_probs_labels[0],
                                                          predicted_probs=predicted_probs_labels[1])
        return fitted_model, metrics

    def _get_feature_matrix(self, pipeline, mode: str = '1D'):
        if mode == '1D':
            feature_matrix = pd.concat(pipeline.value, axis=0)
        else:
            feature_matrix = pd.concat([pd.concat(feature_set, axis=1) for feature_set in pipeline.value], axis=0)
        return feature_matrix
