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

    def _get_feature_matrix(self, list_of_features, mode: str = '1D'):
        if mode == '1D':
            feature_matrix = pd.concat(list_of_features, axis=0)
        else:
            feature_matrix = pd.concat([pd.concat(feature_set, axis=1) for feature_set in list_of_features], axis=0)
        return feature_matrix

    def _init_pipeline_nodes(self, **kwargs):
        if 'feature_generator_type' not in kwargs.keys():
            generator = self.feature_generator_dict['Statistical']
        else:
            generator = self.feature_generator_dict[kwargs['feature_generator_type']]

        feature_extractor = generator(**kwargs['feature_hyperparams'])
        classificator = TimeSeriesClassifier(
            model_hyperparams=kwargs['model_hyperparams'])
        evaluator = PerformanceAnalyzer()

        lambda_func_dict = {'create_list_of_ts': lambda x: ListMonad(*x.values.tolist()),
                            'reduce_basis': lambda x: x[:, 0] if x.shape[1] == 1 else x[:, kwargs['component']],
                            'extract_features': lambda x: feature_extractor.get_features(x),
                            'fit_model': lambda x: classificator.fit(train_features=x, train_target=self.train_target),
                            'predict': lambda x: ListMonad({'predicted_labels': classificator.predict(test_features=x),
                                                            'predicted_probs': classificator.predict_proba(
                                                                test_features=x)}),
                            'evaluate_metrics': lambda MonoidPreds: ListMonad(evaluator.calculate_metrics(
                                target=self.test_target, **MonoidPreds))
                            }

        return feature_extractor, classificator, evaluator, lambda_func_dict
