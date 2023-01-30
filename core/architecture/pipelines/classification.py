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


class ClassificationPipelines:
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
                                             'Reccurence']

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'DataDrivenTSC': self.__ts_data_driven_pipeline,
                         'DataDrivenMultiTSC': self.__multits_data_driven_pipeline,
                         'SpecifiedBasisTSC': self.__specified_basis_pipeline,
                         'SpecifiedFeatureGeneratorTSC': self.__specified_fg_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def __ts_data_driven_pipeline(self, **kwargs):
        data_basis = DataDrivenBasis()
        feature_extractor = StatsRunner(**kwargs['feature_hyperparams'])
        classificator = TimeSeriesClassifier(
            model_hyperparams=kwargs['model_hyperparams'])

        create_list_of_ts = lambda x: ListMonad(*x.values.tolist())
        transform_to_basis = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        reduce_basis = lambda x: x[:, 0] if x.shape[1] == 1 else x[:, kwargs['component']]

        train_pipeline = Right(self.train_features).then(create_list_of_ts).map(transform_to_basis).map(
            reduce_basis).map(feature_extractor.get_features)
        test_pipeline = Right(self.test_features).then(create_list_of_ts).map(transform_to_basis).map(reduce_basis).map(
            feature_extractor.get_features)

        self.basis = data_basis.basis
        train_features = pd.concat(train_pipeline.value, axis=0)
        test_features = pd.concat(test_pipeline.value, axis=0)

        fitted_model = classificator.fit(train_features=train_features,
                                         train_target=self.train_target)
        predicted_probs_labels = (classificator.predict(test_features=test_features),
                                  classificator.predict_proba(test_features=test_features))
        metrics = PerformanceAnalyzer().calculate_metrics(target=self.test_target,
                                                          predicted_labels=predicted_probs_labels[0],
                                                          predicted_probs=predicted_probs_labels[1])

        return classificator, metrics

    def __multits_data_driven_pipeline(self, **kwargs):
        data_basis = DataDrivenBasis()
        feature_extractor = StatsRunner(**kwargs['feature_hyperparams'])
        classificator = TimeSeriesClassifier(model_hyperparams=kwargs['model_hyperparams'])

        create_list_of_ts = lambda x: ListMonad(*x.values.tolist())
        transform_to_basis = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        reduce_basis = lambda list_of_components: ListMonad([component[:, 0] if component.shape[1] == 1 else
                                                             component[:, kwargs['component']]
                                                             for component in list_of_components])
        extract_features = lambda list_of_components: ListMonad([feature_extractor.get_features(component)
                                                                 for component in list_of_components])

        train_pipeline = Right(self.train_features).then(create_list_of_ts).map(transform_to_basis).then(
            reduce_basis).then(extract_features)

        test_pipeline = Right(self.test_features).then(create_list_of_ts).map(transform_to_basis).then(
            reduce_basis).then(extract_features)

        self.basis = data_basis.basis
        train_features = pd.concat([pd.concat(feature_set, axis=1) for feature_set in train_pipeline.value], axis=0)
        test_features = pd.concat([pd.concat(feature_set, axis=1) for feature_set in test_pipeline.value], axis=0)

        fitted_model = classificator.fit(train_features=train_features,
                                         train_target=self.train_target)
        predicted_probs_labels = (classificator.predict(test_features=test_features),
                                  classificator.predict_proba(test_features=test_features))
        metrics = PerformanceAnalyzer().calculate_metrics(target=self.test_target,
                                                          predicted_labels=predicted_probs_labels[0],
                                                          predicted_probs=predicted_probs_labels[1])

        return classificator, metrics

    def __specified_basis_pipeline(self, **kwargs):
        pass

    def __specified_fg_pipeline(self, **kwargs):
        if kwargs['feature_generator_type'] is None:
            generator = self.feature_generator_dict['Statistical']
        else:
            generator = self.feature_generator_dict[kwargs['feature_generator_type']]

        feature_extractor = generator(**kwargs['feature_hyperparams'])
        classificator = TimeSeriesClassifier(
            model_hyperparams=kwargs['model_hyperparams'])

        create_list_of_ts = lambda x: ListMonad(*x.values.tolist()) if kwargs['feature_generator_type'] not \
                                                                       in self.generators_with_matrix_input else x
        fit_model = lambda x: classificator.fit(train_features=x, train_target=self.train_target)
        predict = lambda x: (classificator.predict(test_features=x), classificator.predict_proba(test_features=x))

        fitted_model = Right(self.train_features).then(create_list_of_ts).then(feature_extractor.get_features).then(
            fit_model)
        predicted_probs_labels = Right(self.test_features).then(create_list_of_ts).then(
            feature_extractor.get_features).then(predict)

        metrics = PerformanceAnalyzer().calculate_metrics(target=self.test_target,
                                                          predicted_labels=predicted_probs_labels.value[0],
                                                          predicted_probs=predicted_probs_labels.value[1])

        return classificator, metrics


if __name__ == '__main__':
    # Goes to API
    dataset_list = ['Epilepsy',
                    # 'ECG200',
                    # 'EOGVerticalSignal',
                    # 'DistalPhalanxOutlineCorrect',
                    # 'ScreenType',
                    # 'InlineSkate',
                    # 'ArrowHead'
                    ]
    model_hyperparams = {
        'problem': 'classification',
        'seed': 42,
        'timeout': 4,
        'max_depth': 6,
        'max_arity': 3,
        'cv_folds': 3,
        'logging_level': 20,
        'n_jobs': 4
    }
    feature_hyperparams = {
        'window_mode': True,
        'window_size': 10
    }
    dict_result = {}
    # API workaround
    for dataset_name in dataset_list:
        train, test = DataLoader(dataset_name).load_data()
        pipeline_cls = ClassificationPipelines(train_data=train, test_data=test)
        pipeline_data_driven = ClassificationPipelines(train_data=train, test_data=test)
        # model_1, result_for_1_comp = pipeline(feature_generator_type='Reccurence',
        #                                       model_hyperparams=model_hyperparams,
        #                                       feature_hyperparams=feature_hyperparams)

        model_1, result_for_1_comp = pipeline_data_driven('DataDrivenMultiTSC')(component=0,
                                                                                model_hyperparams=model_hyperparams,
                                                                                feature_hyperparams=feature_hyperparams)
        model_2, result_for_2_comp = pipeline_data_driven('DataDrivenTSC')(component=1,
                                                                           model_hyperparams=model_hyperparams,
                                                                           feature_hyperparams=feature_hyperparams)
        dict_result.update({dataset_name: {'1_component_result': result_for_1_comp,
                                           '2_component_result': result_for_2_comp,
                                           '1_component_model': model_1,
                                           '2_component_model': model_2}})
        pd.DataFrame({'1_component_result': result_for_1_comp,
                      '2_component_result': result_for_2_comp}).to_csv(f'./{dataset_name}.csv')
    _ = 1
