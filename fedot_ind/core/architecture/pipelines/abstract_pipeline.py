from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.metrics.metrics_implementation import RMSE, Accuracy
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader


BENCHMARK = 'M4'


class AbstractPipeline:

    def __init__(self, task, task_params=None):
        self.repo = IndustrialModels().setup_repository()
        self.task = task
        self.task_params = task_params
        _metric_dict = {'classification': Accuracy,
                        'regression': RMSE,
                        'ts_forecasting': RMSE}
        self.base_metric = _metric_dict[self.task]

    def create_pipeline(self, node_list):
        pipeline = PipelineBuilder()
        if isinstance(node_list, dict):
            for branch, nodes in node_list.items():
                if isinstance(branch, int):
                    for node in nodes:
                        pipeline.add_node(node, branch_idx=branch)
                else:
                    pipeline.join_branches(nodes)
        else:
            for node in node_list:
                pipeline.add_node(node)
        return pipeline.build()

    def create_input_data(self, dataset_name):
        if self.task == 'ts_forecasting':
            train_data, _ = DataLoader(
                dataset_name=dataset_name).load_forecast_data(folder=BENCHMARK)
            target = train_data.values[-self.task_params['forecast_length']:].flatten()
            train_data = (train_data, target)
            input_train = DataCheck(
                input_data=train_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
            input_test = None
        else:
            train_data, test_data = DataLoader(
                dataset_name=dataset_name).load_data()
            input_train = DataCheck(
                input_data=train_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
            input_test = DataCheck(
                input_data=train_data,
                task=self.task,
                task_params=self.task_params).check_input_data()
        return input_train, input_test

    def evaluate_pipeline(self, node_list, dataset):
        test_model = self.create_pipeline(node_list)
        train_data, test_data = self.create_input_data(dataset)
        test_model.fit(train_data)
        if self.task == 'ts_forecasting':
            predict = test_model.predict(train_data)
            predict_proba = predict
            target = train_data.features[-self.task_params['forecast_length']:].flatten()
        else:
            predict = test_model.predict(test_data, 'labels')
            predict_proba = test_model.predict(test_data, 'probs')
            target = test_data.target
        metric = self.base_metric(target=target,
                                  predicted_probs=predict_proba.predict,
                                  predicted_labels=predict.predict).metric()

        return dict(fitted_model=test_model,
                    predict_labels=predict.predict,
                    predict_probs=predict_proba.predict,
                    quality_metric=metric)

# class AbstractPipelines:
#     def __init__(self, train_data, test_data):
#         self.train_features = train_data[0]
#         self.train_target = train_data[1]
#         self.test_features = test_data[0]
#         self.test_target = test_data[1]
#         self.basis = None
#
#         self.basis_dict = {i.name: i.value for i in BasisTransformations}
#         self.model_dict = {i.name: i.value for i in MlModel}
#         self.feature_generator_dict = {
#             i.name: i.value for i in FeatureGenerator}
#
#         self.generators_with_matrix_input = ['topological',
#                                              'wavelet',
#                                              'recurrence',
#                                              'quantile']
#
#     def _evaluate(self, classificator, train_features, test_features):
#         fitted_model = classificator.fit(train_features=train_features,
#                                          train_target=self.train_target)
#         predicted_probs_labels = (classificator.predict(test_features=test_features),
#                                   classificator.predict_proba(test_features=test_features))
#         metrics = PerformanceAnalyzer().calculate_metrics(target=self.test_target,
#                                                           predicted_labels=predicted_probs_labels[0],
#                                                           predicted_probs=predicted_probs_labels[1])
#         return fitted_model, metrics
#
#     def get_feature_generator(self, **kwargs):
#         pass
#
#     def _get_feature_matrix(self, list_of_features, mode: str = 'Multi', **kwargs):
#         if mode == '1D':
#             feature_matrix = pd.concat(list_of_features, axis=0)
#             if feature_matrix.shape[0] != len(list_of_features):
#                 feature_matrix = pd.concat(list_of_features, axis=1)
#         elif mode == 'MultiEnsemble':
#             feature_matrix = []
#             for i in range(len(list_of_features[0])):
#                 _ = []
#                 for feature_set in list_of_features:
#                     _.append(feature_set[i])
#                 feature_matrix.append(pd.concat(_, axis=0))
#         elif mode == 'list_of_ts':
#             feature_matrix = []
#             for ts in list_of_features:
#                 list_of_windows = []
#                 for step in range(0, ts.shape[1], kwargs['window_length']):
#                     list_of_windows.append(
#                         ts[:, step:step + kwargs['window_length']])
#                 feature_matrix.append(list_of_windows)
#         else:
#             feature_matrix = pd.concat(
#                 [pd.concat(feature_set, axis=1) for feature_set in list_of_features], axis=0)
#         return feature_matrix
#
#     def _init_pipeline_nodes(self, model_type: str = 'tsc', **kwargs):
#         if 'feature_generator_type' not in kwargs.keys():
#             generator = self.feature_generator_dict['quantile']
#         else:
#             generator = self.feature_generator_dict[kwargs['feature_generator_type']]
#         try:
#             feature_extractor = generator(params=kwargs['feature_hyperparams'])
#
#         except AttributeError:
#             with open(PATH_TO_DEFAULT_PARAMS, 'r') as file:
#                 _feature_gen_params = json.load(file)
#                 params = _feature_gen_params[f'{generator}_extractor']
#             feature_extractor = generator(params)
#         try:
#             classificator = self.model_dict[model_type](model_hyperparams=kwargs['model_hyperparams'],
#                                                         generator_name=kwargs['feature_generator_type'],
#                                                         generator_runner=feature_extractor)
#         except Exception:
#             classificator = None
#
#         lambda_func_dict = {'create_list_of_ts': lambda x: ListMonad(*x.values.tolist()),
#                             'scale': lambda time_series: pd.DataFrame(MinMaxScaler().fit_transform(
#                                 time_series.to_numpy())),
#                             'transpose_matrix': lambda time_series: time_series.T,
#                             'reduce_basis': lambda x: x[:, 0] if x.shape[1] == 1 else x[:, kwargs['component']],
#                             'extract_features': lambda x: feature_extractor.get_features(x),
#                             'fit_model': lambda x: classificator.fit(train_features=x, train_target=self.train_target),
#                             'predict': lambda x: ListMonad({'predicted_labels': classificator.predict(test_features=x),
#                                                             'predicted_probs': classificator.predict_proba(
#                                                                 test_features=x)})
#                             }
#
#         return feature_extractor, classificator, lambda_func_dict
