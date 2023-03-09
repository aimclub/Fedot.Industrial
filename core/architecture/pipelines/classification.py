import pandas as pd
from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.pipelines.abstract_pipeline import AbstractPipelines
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.preprocessing.DatasetLoader import DataLoader
from core.operation.transformation.basis.data_driven import DataDrivenBasis
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
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['extract_features']).then(lambda_func_dict['concatenate_features']).insert(
                self._get_feature_matrix(list_of_features=list_of_features, mode='1D'))
        elif kwargs['pipeline_type'] == 'DataDrivenMultiTSC':
            pipeline = Right(input_data).then(lambda_func_dict['create_list_of_ts']).map(
                lambda_func_dict['transform_to_basis']).then(lambda_func_dict['reduce_basis']).then(
                lambda_func_dict['extract_features']).then(lambda_func_dict['concatenate_features']).insert(
                self._get_feature_matrix(list_of_features=list_of_features, mode=kwargs['ensemble']))
        return pipeline

    def __ts_data_driven_pipeline(self, **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)
        data_basis = DataDrivenBasis()
        data_basis.min_rank = None

        lambda_func_dict['transform_to_basis'] = lambda x: self.basis if self.basis is not None else data_basis.fit(x)
        lambda_func_dict['reduce_basis'] = lambda x: x[:, :data_basis.min_rank] if 'component' not in kwargs.keys() \
            else x[:, kwargs['component']]
        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='DataDrivenTSC')
        classificator_module = train_feature_generator_module.then(lambda_func_dict['fit_model'])

        test_feature_generator_module = self.get_feature_generator(input_data=self.test_features,
                                                                   steps=lambda_func_dict,
                                                                   pipeline_type='DataDrivenTSC')
        prediction = test_feature_generator_module.then(lambda_func_dict['predict'])
        classificator.feature_generator = partial(self.get_feature_generator, steps=lambda_func_dict,
                                                  pipeline_type='DataDrivenTSC')
        self.basis = data_basis.basis
        return classificator, prediction.value[0]

    def __multits_data_driven_pipeline(self, ensemble: str = 'Multi', **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)
        data_basis = DataDrivenBasis()

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
        classificator.feature_generator = partial(self.get_feature_generator, steps=lambda_func_dict,
                                                  pipeline_type='DataDrivenMultiTSC')
        self.basis = data_basis.basis

        return classificator, prediction.value[0]

    def __specified_basis_pipeline(self, **kwargs):
        pass

    def __specified_fg_pipeline(self, **kwargs):
        feature_extractor, classificator, lambda_func_dict = self._init_pipeline_nodes(**kwargs)

        train_feature_generator_module = self.get_feature_generator(input_data=self.train_features,
                                                                    steps=lambda_func_dict,
                                                                    pipeline_type='SpecifiedFeatureGeneratorTSC')
        classificator_module = train_feature_generator_module.then(lambda_func_dict['fit_model'])

        test_feature_generator_module = self.get_feature_generator(input_data=self.test_features,
                                                                   steps=lambda_func_dict,
                                                                   pipeline_type='SpecifiedFeatureGeneratorTSC')
        prediction = test_feature_generator_module.then(lambda_func_dict['predict'])
        classificator.feature_generator = partial(self.get_feature_generator, steps=lambda_func_dict,
                                                  pipeline_type='SpecifiedFeatureGeneratorTSC')
        return classificator, prediction.value[0]


if __name__ == "__main__":
    dataset_list = [
        # 'BirdChicken', win by statistical 5% window
        # 'Computers',
        # 'DistalPhalanxOutlineCorrect',
        'ECG200',
        # 'FordA',
        'GunPointAgeSpan',
        'Herring',
        'Lightning2',
        'MiddlePhalanxOutlineCorrect',
        'MoteStrain',
        'PhalangesOutlines Correct',
        'ProximalPhalanxOutlineCorrect',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'Strawberry',
        'ToeSegmentation2',
        'TwoLegECG',
        'WormsTwoClass',
        'Yoga']
    dataset_list = [
        # 'ACSF1',
        #             'Adiac',
        #             'ArrowHead',
        #             'ChlorineConcentration',
        #             'CricketX',
        #             'CricketY',
        #             'CricketZ',
        #             'DistalPhalanxTW',
        #             'DistalPhalanxOutlineAgeGroup',
        #             'ECG5000',
        #             'ElectricDevices',
        #             'EOGVerticalSignal',
        #             'EthanolLevel',
        #             'FaceFour',
        #             'Haptics',
        #             'InlineSkate',
        #             'LargeKitchenAppliances',
        #             'Lightning7',
                    #'Mallat',
                    #'Meat',
                    # 'MiddlePhalanxOutlineAgeGroup',
                    # 'MiddlePhalanxTW',
                    #'OliveOil',
                    # 'Phoneme',
                    # 'RefrigerationDevices',
                    #'ScreenType',
                    'SwedishLeaf',
                    ]
    model_hyperparams = {
        'problem': 'classification',
        'seed': 42,
        'metric': 'f1',
        'timeout': 7,
        'max_depth': 6,
        'max_arity': 3,
        'cv_folds': 3,
        'logging_level': 20,
        'n_jobs': 4
    }
    # df = pd.read_csv('./results_topo.csv')
    dict_result = {}

    for dataset_name in dataset_list:
        try:
            train, test = DataLoader(dataset_name).load_data()
            pipeline = ClassificationPipelines(train_data=train, test_data=test)
            window_result = {}
            for window_size in [5, 10, 20, 30, 50]:
                feature_hyperparams = {
                    'window_mode': True,
                    'window_size': window_size
                }
                model_1, result_for_1_comp = pipeline('SpecifiedFeatureGeneratorTSC')(
                    feature_generator_type='statistical',
                    model_hyperparams=model_hyperparams,
                    feature_hyperparams=feature_hyperparams)
                # model_1, result_for_1_comp = pipeline('DataDrivenTSC')(model_hyperparams=model_hyperparams,
                #                                                        feature_hyperparams=feature_hyperparams)
                metrics = PerformanceAnalyzer().calculate_metrics(target=test[1],
                                                                  predicted_labels=result_for_1_comp[
                                                                      'predicted_labels'],
                                                                  predicted_probs=result_for_1_comp['predicted_probs'])
                window_result.update({f'{window_size}': {'model': (model_1, result_for_1_comp),
                                                         'metrics': metrics}})
                print(dataset_name)
                print(window_size)
                print(metrics)

            dict_result.update({dataset_name: window_result})
            pd.DataFrame(dict_result).to_csv('./results_multi_stat.csv')
        except Exception:
            print('Lel')
    _ = 1
    # model_1, result_for_1_comp = pipeline_data_driven('DataDrivenMultiTSC')(component=[0, 0, 0],
    #                                                                         ensemble='MultiEnsemble',
    #                                                                         model_hyperparams=model_hyperparams,
    #                                                                         feature_hyperparams=feature_hyperparams)
