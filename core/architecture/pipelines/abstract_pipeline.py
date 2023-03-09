import pandas as pd
from pymonad.list import ListMonad
from core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from core.architecture.settings.pipeline_settings import BasisTransformations, FeatureGenerator, MlModel


class AbstractPipelines:
    def __init__(self, train_data, test_data):
        self.train_features = train_data[0]
        self.train_target = train_data[1]
        self.test_features = test_data[0]
        self.test_target = test_data[1]
        self.basis = None

        self.basis_dict = {i.name: i.value for i in BasisTransformations}
        self.model_dict = {i.name: i.value for i in MlModel}
        self.feature_generator_dict = {i.name: i.value for i in FeatureGenerator}

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

    def get_feature_generator(self, **kwargs):
        pass

    def _get_feature_matrix(self, list_of_features, mode: str = 'Multi'):
        if mode == '1D':
            feature_matrix = pd.concat(list_of_features, axis=0)
            if feature_matrix.shape[1] != len(list_of_features):
                feature_matrix = pd.concat(list_of_features, axis=1)
        elif mode == 'MultiEnsemble':
            feature_matrix = []
            for i in range(len(list_of_features[0])):
                _ = []
                for feature_set in list_of_features:
                    _.append(feature_set[i])
                feature_matrix.append(pd.concat(_, axis=0))
        else:
            feature_matrix = pd.concat([pd.concat(feature_set, axis=1) for feature_set in list_of_features], axis=0)
        return feature_matrix

    def _init_pipeline_nodes(self, model_type: str = 'tsc', **kwargs):
        if 'feature_generator_type' not in kwargs.keys():
            generator = self.feature_generator_dict['statistical']
        else:
            generator = self.feature_generator_dict[kwargs['feature_generator_type']]

        feature_extractor = generator(**kwargs['feature_hyperparams'])
        classificator = self.model_dict[model_type](model_hyperparams=kwargs['model_hyperparams'])
    # TODO:
        lambda_func_dict = {'create_list_of_ts': lambda x: ListMonad(*x.values.tolist()),
                            'reduce_basis': lambda x: x[:, 0] if x.shape[1] == 1 else x[:, kwargs['component']],
                            'extract_features': lambda x: feature_extractor.get_features(x),
                            'fit_model': lambda x: classificator.fit(train_features=x, train_target=self.train_target),
                            'predict': lambda x: ListMonad({'predicted_labels': classificator.predict(test_features=x),
                                                            'predicted_probs': classificator.predict_proba(
                                                                test_features=x)})
                            }

        return feature_extractor, classificator, lambda_func_dict
    # API workaround
    # train = pd.read_csv(
    #     r'D:\РАБОТЫ РЕПОЗИТОРИИ\Репозитории\Industiral\IndustrialTS\data\FedotMedical\FedotMedical_TRAIN.tsv',
    #     sep=',')
    # df_1_target = train[train['target'] == 1]
    # df_0_target = train[train['target'] == 0].sample(frac=0.3)
    # train = pd.concat([df_0_target, df_1_target], axis=0)
    # target = train['target'].values
    # train_ts = train[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6',
    #                   'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature12',
    #                   'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18',
    #                   'Feature19', 'Feature20', 'Feature21', 'Feature22', 'Feature23', 'Feature24']]
    # train_binary = train[['Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
    #                       'Feature32', 'Feature33', 'Feature34', 'Feature35', 'Feature36', 'Feature37',
    #                       'Feature38', 'Feature39', 'Feature40', 'Feature41', 'Feature42', 'Feature43',
    #                       'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49',
    #                       'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
    #                       'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61']]
    # percent_missing = train_ts.isnull().sum() * 100 / len(train)
    # missing_value_df = pd.DataFrame({'column_name': train_ts.columns,
    #                                  'percent_missing': percent_missing})
    # train_ts = train_ts[['Feature1', 'Feature2', 'Feature4', 'Feature5', 'Feature6', 'Feature22']]
    # percent_missing = train_binary.isnull().sum() * 100 / len(train)
    # missing_value_df = pd.DataFrame({'column_name': train_binary.columns,
    #                                  'percent_missing': percent_missing})
    # binary_train, binary_test, binary_train_target, binary_test_target = train_test_split(train_binary.values,
    #                                                                                       target,
    #                                                                                       stratify=target,
    #                                                                                       test_size=0.25)
    # binary_clf = TimeSeriesClassifier(model_hyperparams=model_hyperparams)
    # binary_clf = binary_clf.fit(train_features=binary_train, train_target=binary_train_target)
    # preds = binary_clf.predict(features=binary_test)
    # preds_proba = binary_clf.predict_proba(features=binary_test)
    # metrics = PerformanceAnalyzer().calculate_metrics(
    #     target=binary_test_target,
    #     predicted_labels=preds,
    #     predicted_probs=preds_proba)
    # X_train, X_test, y_train, y_test = train_test_split(train_ts.values, target,
    #                                                     stratify=target,
    #                                                     test_size=0.25)
    # train, test = (X_train, y_train), (X_test, y_test)
    # pipeline_cls = ClassificationPipelines(train_data=train, test_data=test)
    # model_1, result_for_1_comp = pipeline_cls('SpecifiedFeatureGeneratorTSC')(feature_generator_type='Statistical',
    #                                                                           model_hyperparams=model_hyperparams,
    #                                                                           feature_hyperparams=feature_hyperparams)
    # test_features = model_1.test_features
    # probs_ts = model_1.predict_proba(test_features)
    # label_ts = model_1.predict(test_features)
    # ensemble_metric = []
    # ensemble_probs = np.concatenate([probs_ts, preds_proba], axis=1)
    # for method in [np.mean, np.max, np.min, np.sum]:
    #     probs = method(ensemble_probs, axis=1)
    #     label = np.round(probs)
    #     metric_by_ensemble = PerformanceAnalyzer().calculate_metrics(
    #         target=binary_test_target,
    #         predicted_labels=preds,
    #         predicted_probs=probs)
    #     ensemble_metric.append(metric_by_ensemble)
