from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.pipelines.abstract_pipeline import AbstractPipelines
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.operation.transformation.basis.data_driven import DataDrivenBasisImplementation
from functools import partial


class ClassificationPipelines(AbstractPipelines):

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'DataDrivenTSC': self.__ts_data_driven_pipeline,
                         'DataDrivenMultiTSC': self.__multits_data_driven_pipeline,
                         'SpecifiedBasisTSC': self.__specified_basis_pipeline,
                         'SpecifiedFeatureGeneratorTSC': self.__specified_fg_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def get_feature_generator(self, **kwargs):
        lambda_func_dict = kwargs['steps']
        lambda_func_dict['concatenate_features'] = lambda x: list_of_features.append(x)
        input_data = kwargs['input_data']
        list_of_features = []
        if kwargs['pipeline_type'] == 'DataDrivenTSC':
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['transform_to_basis']).map(lambda_func_dict['reduce_basis']).map(
                lambda_func_dict['extract_features']).then(lambda_func_dict['concatenate_features']).insert(
                self._get_feature_matrix(list_of_features=list_of_features, mode='1D'))
        elif kwargs['pipeline_type'] == 'SpecifiedFeatureGeneratorTSC':
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).then(
                lambda_func_dict['extract_features'])
        elif kwargs['pipeline_type'] == 'DataDrivenMultiTSC':
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['transform_to_basis']).then(lambda_func_dict['reduce_basis']).then(
                lambda_func_dict['extract_features']).then(lambda_func_dict['concatenate_features']).insert(
                self._get_feature_matrix(list_of_features=list_of_features, mode=kwargs['ensemble']))
        return pipeline

    def __ts_data_driven_pipeline(self, **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)
        data_basis = DataDrivenBasisImplementation()

        lambda_func_dict['transform_to_basis'] = lambda x: self.basis if self.basis is not None else data_basis.fit(x)

        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='DataDrivenTSC')
        classificator_module = train_feature_generator_module.then(lambda_func_dict['fit_model'])

        test_feature_generator_module = self.get_feature_generator(input_data=self.test_features,
                                                                   steps=lambda_func_dict,
                                                                   pipeline_type='DataDrivenTSC')
        prediction = test_feature_generator_module.then(lambda_func_dict['predict'])

        self.basis = data_basis.basis
        return classificator, prediction.value[0]

    def __multits_data_driven_pipeline(self, ensemble: str = 'Multi', **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)
        data_basis = DataDrivenBasisImplementation()

        lambda_func_dict['transform_to_basis'] = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        lambda_func_dict['reduce_basis'] = lambda list_of_components: ListMonad(
            [component[:, 0] if component.shape[1] == 1 else
             component[:, rank]
             for component, rank in zip(list_of_components, kwargs['component'])])
        lambda_func_dict['extract_features'] = lambda list_of_components: ListMonad(
            [feature_extractor.get_features(component)
             for component in list_of_components])

        if ensemble == 'MultiEnsemble':
            lambda_func_dict['fit_model'] = lambda feature_by_each_component: ListMonad({str(index):
                classificator.fit(
                    train_features=feature_set, train_target=self.train_target) for index, feature_set in
                enumerate(feature_by_each_component)})
            lambda_func_dict['predict'] = lambda feature_by_each_component: ListMonad(
                {index: {'predicted_labels': classificator.predict(features=feature_set),
                         'predicted_probs': classificator.predict_proba(
                             features=feature_set)} for (index, classificator), feature_set in
                 zip(classificator_module.value[0].items(), feature_by_each_component)})

        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='DataDrivenMultiTSC',
                                                                    ensemble=ensemble)

        classificator_module = train_feature_generator_module.then(lambda_func_dict['fit_model'])

        test_feature_generator_module = self.get_feature_generator(input_data=self.test_features,
                                                                   steps=lambda_func_dict,
                                                                   pipeline_type='DataDrivenMultiTSC',
                                                                   ensemble=ensemble)
        prediction = test_feature_generator_module.then(lambda_func_dict['predict'])

        self.basis = data_basis.basis

        return classificator, prediction.value[0]

    def __specified_basis_pipeline(self, **kwargs):
        pass

    def __specified_fg_pipeline(self, **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)

        lambda_func_dict['create_list_of_ts'] = lambda x: ListMonad(*x.values.tolist()) if \
            kwargs['feature_generator_type'] not in self.generators_with_matrix_input else x
        lambda_func_dict['extract_features'] = lambda x: ListMonad(feature_extractor.get_features(x))

        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='SpecifiedFeatureGeneratorTSC')
        classificator_module = train_feature_generator_module.then(lambda_func_dict['fit_model'])

        test_feature_generator_module = self.get_feature_generator(input_data=self.test_features,
                                                                   steps=lambda_func_dict,
                                                                   pipeline_type='SpecifiedFeatureGeneratorTSC')
        prediction = test_feature_generator_module.then(lambda_func_dict['predict'])

        return classificator, prediction.value[0]
