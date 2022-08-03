import timeit

from gtda.time_series import SingleTakensEmbedding

from cases.run.ExperimentRunner import ExperimentRunner
from core.models.topological.TDA import Topological
from core.models.topological.external.TFE import TopologicalFeaturesExtractor, PersistenceDiagramsExtractor, \
    HolesNumberFeature, MaxHoleLifeTimeFeature, RelevantHolesNumber, AverageHoleLifetimeFeature, \
    SumHoleLifetimeFeature, PersistenceEntropyFeature, SimultaneousAliveHolesFeature, \
    AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, RadiusAtMaxBNFeature
from core.operation.utils.utils import *

# from multiprocessing.dummy import Pool
# from fedot.api.main import Fedot
# from fedot.core.data.data import InputData
# from fedot.core.data.supplementary_data import SupplementaryData
# from fedot.core.repository.dataset_types import DataTypesEnum
# from fedot.core.repository.tasks import TaskTypesEnum, Task
# from sklearn.model_selection import train_test_split

dict_of_dataset = dict
dict_of_win_list = dict

PERSISTENCE_DIAGRAM_FEATURES = {'HolesNumberFeature': HolesNumberFeature(),
                                'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
                                'RelevantHolesNumber': RelevantHolesNumber(),
                                'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
                                'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
                                'PersistenceEntropyFeature': PersistenceEntropyFeature(),
                                'SimultaneousAliveHolesFeature': SimultaneousAliveHolesFeature(),
                                'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
                                'BettiNumbersSumFeature': BettiNumbersSumFeature(),
                                'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()}


class TopologicalRunner(ExperimentRunner):
    def __init__(self, topological_params: dict,
                 list_of_dataset: list = None):
        super().__init__(list_of_dataset)
        self.topological_extractor = Topological(**topological_params)
        self.TE_dimension = None
        self.TE_time_delay = None

    def generate_topological_features(self, ts_data: pd.DataFrame):
        start = timeit.default_timer()

        if not self.TE_dimension and not self.TE_time_delay:
            single_ts = ts_data.loc[0]
            self.TE_dimension, self.TE_time_delay = self.get_embedding_params(single_time_series=single_ts)
            self.logger.info(f'TE_delay: {self.TE_time_delay}, TE_dimension: {self.TE_dimension} are selected')

        persistence_diagram_extractor = PersistenceDiagramsExtractor(takens_embedding_dim=self.TE_dimension,
                                                                     takens_embedding_delay=self.TE_time_delay,
                                                                     homology_dimensions=(0, 1),
                                                                     parallel=True)

        feature_extractor = TopologicalFeaturesExtractor(persistence_diagram_extractor=persistence_diagram_extractor,
                                                         persistence_diagram_features=PERSISTENCE_DIAGRAM_FEATURES)

        ts_data_transformed = feature_extractor.fit_transform(ts_data.values)
        ts_data_transformed = delete_col_by_var(ts_data_transformed)

        time_elapsed = round(timeit.default_timer() - start, 2)
        self.logger.info(f'Time spent on feature generation - {time_elapsed} sec')
        return ts_data_transformed

    def extract_features(self, ts_data: pd.DataFrame, dataset_name: str = None):
        self.logger.info('Topological features extraction started')
        return self.generate_topological_features(ts_data=ts_data)

    @staticmethod
    def get_embedding_params(single_time_series):
        embedder = SingleTakensEmbedding(parameters_type="search",
                                         time_delay=10,
                                         dimension=10)
        embedder.fit_transform(single_time_series)
        return embedder.dimension_, embedder.time_delay_

    # def generate_features_from_ts(self, ts_frame, window_length=None):
    #     pool = Pool(8)
    #     start = timeit.default_timer()
    #     self.ts_samples_count = ts_frame.shape[0]
    #     self.topological_extractor.time_series = ts_frame
    #     aggregation_df = pool.map(self.topological_extractor.time_series_rolling_betti_ripser, ts_frame.values)
    #     betti_sum = [x['Betti_sum'].values.tolist() for x in aggregation_df]
    #     feats = pd.DataFrame(betti_sum)
    #     pool.close()
    #     pool.join()
    #     self.logger.info(f'Time spent on feature generation - {round((timeit.default_timer() - start), 2)} sec')
    #     return feats, aggregation_df

    # @staticmethod
    # def __convert_to_input_data(features,
    #                             target,
    #                             col_numbers):
    #     converted_features = InputData(idx=np.arange(len(features)),
    #                                    features=features,
    #                                    target=target,
    #                                    task=Task(TaskTypesEnum.classification),
    #                                    data_type=DataTypesEnum.table,
    #                                    supplementary_data=SupplementaryData(was_preprocessed=True,
    #                                                                         column_types={
    #                                                                             'features': [float] * col_numbers,
    #                                                                             'target': [int]}))
    #     return converted_features

    # def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length: int = None):
    #
    #     self.logger.info('Generating features for fit model')
    #     self.window_length = window_length
    #     self.logger.info('Start fitting FEDOT model')
    #
    #     converted_features = self.__convert_to_input_data(features=X_train,
    #                                                       target=y_train,
    #                                                       col_numbers=X_train.shape[1])
    #
    #     predictor = Fedot(**self.fedot_params)
    #     predictor.fit(features=X_train, target=y_train)
    #     return predictor
    #
    # def _predict_on_train(self, predictor):
    #
    #     # Predict on whole TRAIN
    #     predictions, predictions_proba, inference = self.predict(predictor=predictor,
    #                                                              X_test=self.train_feats,
    #                                                              window_length=self.window_length,
    #                                                              y_test=self.y_train)
    #
    #     # GEt metrics on TRAIN
    #     metrics = self.analyzer.calculate_metrics(self.metrics_name,
    #                                               target=self.y_train,
    #                                               predicted_labels=predictions,
    #                                               predicted_probs=predictions_proba
    #                                               )
    #
    #     return dict(predictions=predictions,
    #                 predictions_proba=predictions_proba,
    #                 inference=inference,
    #                 metrics=metrics)
    #
    # def predict(self, predictor,
    #             X_test: pd.DataFrame,
    #             window_length: int = None,
    #             y_test: np.array = None):
    #     self.logger.info('Generating features for prediction')
    #
    #     converted_features = self.__convert_to_input_data(features=X_test,
    #                                                       target=y_test,
    #                                                       col_numbers=X_test.shape[1])
    #
    #     start_time = timeit.default_timer()
    #     predictions = predictor.predict(features=X_test)
    #     inference = timeit.default_timer() - start_time
    #     predictions_proba = predictor.predict_proba(features=X_test)
    #
    #     return predictions, predictions_proba, inference

    # def run_experiment(self,
    #                    method,
    #                    dict_of_dataset: dict,
    #                    dict_of_win_list: dict,
    #                    save_features: bool = False,
    #                    single_window_mode: bool = True):
    #     for dataset in self.list_of_dataset:
    #         trajectory_windows_list = dict_of_win_list[dataset]
    #         for launch in range(self.launches):
    #             try:
    #                 self.path_to_save = self._create_path_to_save(method, dataset, launch)
    #                 X, y = dict_of_dataset[dataset]
    #
    #                 if type(X) is tuple:
    #                     self.X_train, self.X_test, self.y_train, self.y_test = X[0], X[1], y[0], y[1]
    #                 else:
    #                     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))
    #
    #                 self._get_clf_params()
    #
    #                 if single_window_mode:
    #                     self.window_length = trajectory_windows_list[launch]
    #                     window_length_list = trajectory_windows_list[launch]
    #                     self.logger.info('Generate pipeline for trajectory matrix with window length - {}'.format(
    #                         self.window_length))
    #                     self.test_feats = None
    #                     self.train_feats = None
    #                 else:
    #                     window_length_list = trajectory_windows_list
    #
    #                 self.train_feats, self.test_feats = self.generate_topological_features(X_train=self.X_train,
    #                                                                                        X_test=self.X_test,
    #                                                                                        takens_embedding_delay=self.window_length)
    #                 predictor = self.fit(X_train=self.train_feats,
    #                                      y_train=self.y_train,
    #                                      window_length=self.window_length)
    #
    #                 result_on_train = self._predict_on_train(predictor=predictor)
    #
    #                 self.X_test = self.test_feats
    #
    #                 self._get_dimension_params(predictions_proba_train=result_on_train['predictions_proba'])
    #
    #                 result_on_test = self._predict_on_test(predictor=predictor)
    #
    #                 if not os.path.exists(self.path_to_save):
    #                     os.makedirs(self.path_to_save)
    #
    #                 if save_features:
    #                     pd.DataFrame(self.train_feats).to_csv(os.path.join(self.path_to_save, 'train_features.csv'))
    #                     pd.DataFrame(self.y_train).to_csv(os.path.join(self.path_to_save, 'train_target.csv'))
    #                     pd.DataFrame(self.test_feats).to_csv(os.path.join(self.path_to_save, 'test_features.csv'))
    #                     pd.DataFrame(self.y_test).to_csv(os.path.join(self.path_to_save, 'test_target.csv'))
    #
    #                 self._save_all_results(predictor=predictor,
    #                                        boosting_results=None,
    #                                        normal_results=result_on_test)
    #
    # # boosting_results = self._predict_with_boosting(predictions=result_on_test['predictions'], #
    # predictions_proba=result_on_test[ #                                                    'predictions_proba'],
    #                                                metrics_without_boosting=result_on_test['metrics']) #
    #                                                boosting_results['dataset'] = dataset # #
    #                                                self._save_all_results(predictor=predictor, #
    #                                                boosting_results=boosting_results, #
    #                                                normal_results=result_on_test, #
    #                                                save_boosting=True)
    #
    #             except Exception as ex:
    #                 print(ex)
    #                 print(str(dataset))
