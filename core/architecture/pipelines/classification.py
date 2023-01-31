from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.pipelines.abstract_pipeline import AbstractPipelines
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.operation.transformation.basis.data_driven import DataDrivenBasis


class ClassificationPipelines(AbstractPipelines):

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'DataDrivenTSC': self.__ts_data_driven_pipeline,
                         'DataDrivenMultiTSC': self.__multits_data_driven_pipeline,
                         'SpecifiedBasisTSC': self.__specified_basis_pipeline,
                         'SpecifiedFeatureGeneratorTSC': self.__specified_fg_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def __ts_data_driven_pipeline(self, **kwargs):
        feature_extractor, classificator, evaluator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)

        data_basis = DataDrivenBasis()
        list_of_features = []

        transform_to_basis = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        concatenate_features = lambda x: list_of_features.append(x)

        train_pipeline = Right(self.train_features).then(lambda_func_dict['create_list_of_ts']).map(
            transform_to_basis).map(lambda_func_dict['reduce_basis']).map(lambda_func_dict['extract_features']).then(
            concatenate_features).insert(self._get_feature_matrix(list_of_features=list_of_features, mode='1D')).then(
            lambda_func_dict['fit_model'])

        list_of_features = []
        test_pipeline = Right(self.test_features).then(lambda_func_dict['create_list_of_ts']).map(
            transform_to_basis).map(lambda_func_dict['reduce_basis']).map(lambda_func_dict['extract_features']).then(
            concatenate_features).insert(self._get_feature_matrix(list_of_features=list_of_features, mode='1D')).then(
            lambda_func_dict['predict']).then(lambda_func_dict['evaluate_metrics'])

        self.basis = data_basis.basis
        return classificator, test_pipeline.value

    def __multits_data_driven_pipeline(self, ensemble: str = 'Multi', **kwargs):
        feature_extractor, classificator, evaluator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)

        data_basis = DataDrivenBasis()
        list_of_features = []

        transform_to_basis = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        lambda_func_dict['reduce_basis'] = lambda list_of_components: ListMonad(
            [component[:, 0] if component.shape[1] == 1 else
             component[:, rank]
             for component, rank in zip(list_of_components, kwargs['component'])])
        lambda_func_dict['extract_features'] = lambda list_of_components: ListMonad(
            [feature_extractor.get_features(component)
             for component in list_of_components])
        concatenate_features = lambda x: list_of_features.append(x)

        if ensemble == 'MultiEnsemble':
            lambda_func_dict['fit_model'] = lambda feature_by_each_component: ListMonad({str(index):
                classificator.fit(
                    train_features=feature_set, train_target=self.train_target) for index, feature_set in
                enumerate(feature_by_each_component)})
            lambda_func_dict['predict'] = lambda feature_by_each_component: ListMonad(
                {index: {'predicted_labels': classificator.predict(features=feature_set),
                         'predicted_probs': classificator.predict_proba(
                             features=feature_set)} for (index, classificator), feature_set in
                 zip(train_pipeline.value[0].items(), feature_by_each_component)})
            lambda_func_dict['evaluate'] = lambda PredDict: ListMonad({index: evaluator.calculate_metrics(
                target=self.test_target, **preds) for index, preds in PredDict.items()})

        train_pipeline = Right(self.train_features).then(lambda_func_dict['create_list_of_ts']).map(
            transform_to_basis).then(lambda_func_dict['reduce_basis']).then(lambda_func_dict['extract_features']).then(
            concatenate_features).insert(
            self._get_feature_matrix(list_of_features=list_of_features, mode=ensemble)).then(
            lambda_func_dict['fit_model'])

        list_of_features = []
        test_pipeline = Right(self.test_features).then(lambda_func_dict['create_list_of_ts']).map(
            transform_to_basis).then(lambda_func_dict['reduce_basis']).then(lambda_func_dict['extract_features']).then(
            concatenate_features).insert(
            self._get_feature_matrix(list_of_features=list_of_features, mode=ensemble)).then(
            lambda_func_dict['predict']).then(lambda_func_dict['evaluate_metrics'])

        self.basis = data_basis.basis

        return classificator, test_pipeline.value

    def __specified_basis_pipeline(self, **kwargs):
        pass

    def __specified_fg_pipeline(self, **kwargs):
        feature_extractor, classificator, evaluator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)

        lambda_func_dict['create_list_of_ts'] = lambda x: ListMonad(*x.values.tolist()) if \
            kwargs['feature_generator_type'] not in self.generators_with_matrix_input else x
        lambda_func_dict['extract_features'] = lambda x: ListMonad(feature_extractor.get_features(x))

        fitted_model = Right(self.train_features).then(lambda_func_dict['create_list_of_ts']).then(
            lambda_func_dict['extract_features']).then(lambda_func_dict['fit_model'])
        predicted_probs_labels = Right(self.test_features).then(lambda_func_dict['create_list_of_ts']).then(
            lambda_func_dict['extract_features']).then(lambda_func_dict['predict']).then(
            lambda_func_dict['evaluate_metrics'])

        return classificator, predicted_probs_labels.value


if __name__ == '__main__':
    # Goes to API
    dataset_list = [
        'Epilepsy',
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
        'timeout': 2,
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
        # model_1, result_for_1_comp = pipeline_cls('SpecifiedFeatureGeneratorTSC')(feature_generator_type='Statistical',
        #                                                                           model_hyperparams=model_hyperparams,
        #                                                                           feature_hyperparams=feature_hyperparams)
        #
        model_1, result_for_1_comp = pipeline_data_driven('DataDrivenMultiTSC')(component=[0, 0, 0],
                                                                                ensemble='MultiEnsemble',
                                                                                model_hyperparams=model_hyperparams,
                                                                                feature_hyperparams=feature_hyperparams)
        # model_2, result_for_2_comp = pipeline_data_driven('DataDrivenTSC')(component=0,
        #                                                                    model_hyperparams=model_hyperparams,
        #                                                                    feature_hyperparams=feature_hyperparams)
        # dict_result.update({dataset_name: {'1_component_result': result_for_1_comp,
        #                                    '2_component_result': result_for_2_comp,
        #                                    '1_component_model': model_1,
        #                                    '2_component_model': model_2}})
        # pd.DataFrame({'1_component_result': result_for_1_comp,
        #               '2_component_result': result_for_2_comp}).to_csv(f'./{dataset_name}.csv')
    _ = 1
