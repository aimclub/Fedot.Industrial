from multiprocessing.dummy import Pool
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
from core.models.topological.external import *
from core.models.topological.TDA import Topological
from cases.run.ExperimentRunner import ExperimentRunner
from core.models.topological.external.TFE import TopologicalFeaturesExtractor, PersistenceDiagramsExtractor, \
    HolesNumberFeature, MaxHoleLifeTimeFeature, RelevantHolesNumber, AverageHoleLifetimeFeature, SumHoleLifetimeFeature, \
    PersistenceEntropyFeature, SimultaneousAliveHolesFeatue, AveragePersistenceLandscapeFeature, BettiNumbersSumFeature, \
    RadiusAtMaxBNFeature
from core.operation.utils.utils import *
import timeit

dict_of_dataset = dict
dict_of_win_list = dict


class TopologicalRunner(ExperimentRunner):
    def __init__(self,
                 topological_params: dict,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = None
                 ):

        super().__init__(list_of_dataset, launches, metrics_name, fedot_params)
        self.topological_extractor = Topological(**topological_params)

    def __convert_to_input_data(self, features,
                                target,
                                col_numbers):
        converted_features = InputData(idx=np.arange(len(features)),
                                       features=features,
                                       target=target,
                                       task=Task(TaskTypesEnum.classification),
                                       data_type=DataTypesEnum.table,
                                       supplementary_data=SupplementaryData(was_preprocessed=True,
                                                                            column_types={
                                                                                'features': [float] * col_numbers,
                                                                                'target': [int]}))
        return converted_features

    def generate_features_from_ts(self, ts_frame, window_length=None):
        pool = Pool(8)
        start = timeit.default_timer()
        self.ts_samples_count = ts_frame.shape[0]
        self.topological_extractor.time_series = ts_frame
        aggregation_df = pool.map(self.topological_extractor.time_series_rolling_betti_ripser, ts_frame.values)
        betti_sum = [x['Betti_sum'].values.tolist() for x in aggregation_df]
        feats = pd.DataFrame(betti_sum)
        pool.close()
        pool.join()
        self.logger.info(f'Time spent on feature generation - {timeit.default_timer() - start}')
        return feats, aggregation_df

    def generate_topological_features(self, X_train, X_test, tokens_embedding_delay):
        feature_extractor = TopologicalFeaturesExtractor(
            persistence_diagram_extractor=PersistenceDiagramsExtractor(tokens_embedding_dim=2,
                                                                       tokens_embedding_delay=tokens_embedding_delay,
                                                                       homology_dimensions=(0, 1),
                                                                       parallel=True),
            persistence_diagram_features={'HolesNumberFeature': HolesNumberFeature(),
                                          'MaxHoleLifeTimeFeature': MaxHoleLifeTimeFeature(),
                                          'RelevantHolesNumber': RelevantHolesNumber(),
                                          'AverageHoleLifetimeFeature': AverageHoleLifetimeFeature(),
                                          'SumHoleLifetimeFeature': SumHoleLifetimeFeature(),
                                          'PersistenceEntropyFeature': PersistenceEntropyFeature(),
                                          'SimultaneousAliveHolesFeatue': SimultaneousAliveHolesFeatue(),
                                          'AveragePersistenceLandscapeFeature': AveragePersistenceLandscapeFeature(),
                                          'BettiNumbersSumFeature': BettiNumbersSumFeature(),
                                          'RadiusAtMaxBNFeature': RadiusAtMaxBNFeature()})

        X_train_transformed = feature_extractor.fit_transform(X_train.values)
        X_test_transformed = feature_extractor.fit_transform(X_test.values)
        X_train_transformed = delete_col_by_var(X_train_transformed)
        X_test_transformed = delete_col_by_var(X_test_transformed)
        return X_train_transformed, X_test_transformed

    def _generate_fit_time(self, predictor):
        fit_time = []
        if predictor.best_models is None:
            fit_time.append(predictor.current_pipeline.computation_time)
        else:
            for model in predictor.best_models:
                current_computation = model.computation_time
                fit_time.append(current_computation)
        return fit_time

    def _create_path_to_save(self, method,dataset, launch):
        save_path = os.path.join(path_to_save_results(),method, dataset, str(launch))
        return save_path

    def extract_features(self,
                         dataset,
                         dict_of_dataset,
                         dict_of_extra_params):
        X_train, X_test, y_train, y_test = self._load_data(dataset=dataset, dict_of_dataset=dict_of_dataset)
        return self.generate_topological_features(X_train, X_test, dict_of_extra_params[dataset])

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length: int = None):

        self.logger.info('Generating features for fit model')
        self.window_length = window_length
        # train_feats, betti_df_train = self.generate_features_from_ts(X_train)

        self.logger.info('Start fitting FEDOT model')

        converted_features = self.__convert_to_input_data(features=X_train,
                                                          target=y_train,
                                                          col_numbers=X_train.shape[1])

        predictor = Fedot(**self.fedot_params)
        predictor.fit(features=X_train, target=y_train)
        return predictor

    def _predict_on_train(self, predictor):

        # Predict on whole TRAIN
        predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                 X_test=self.train_feats,
                                                                 window_length=self.window_length,
                                                                 y_test=self.y_train)

        # GEt metrics on TRAIN
        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                  target=self.y_train,
                                                  predicted_labels=predictions,
                                                  predicted_probs=predictions_proba
                                                  )

        return dict(predictions=predictions,
                    predictions_proba=predictions_proba,
                    inference=inference,
                    metrics=metrics)

    def predict(self, predictor,
                X_test: pd.DataFrame,
                window_length: int = None,
                y_test: np.array = None):
        self.logger.info('Generating features for prediction')

        # test_feats, betti_df_test = self.generate_features_from_ts(ts_frame=X_test, window_length=window_length)

        converted_features = self.__convert_to_input_data(features=X_test,
                                                          target=y_test,
                                                          col_numbers=X_test.shape[1])

        start_time = timeit.default_timer()
        predictions = predictor.predict(features=X_test)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=X_test)

        return predictions, predictions_proba, inference

    def run_experiment(self,
                       method,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict,
                       save_features: bool = False,
                       single_window_mode: bool = True):
        for dataset in self.list_of_dataset:
            trajectory_windows_list = dict_of_win_list[dataset]
            for launch in range(self.launches):
                try:
                    self.path_to_save = self._create_path_to_save(method, dataset, launch)
                    X, y = dict_of_dataset[dataset]

                    if type(X) is tuple:
                        self.X_train, self.X_test, self.y_train, self.y_test = X[0], X[1], y[0], y[1]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))

                    self._get_clf_params()

                    if single_window_mode:
                        self.window_length = trajectory_windows_list[launch]
                        window_length_list = trajectory_windows_list[launch]
                        self.logger.info('Generate pipeline for trajectory matrix with window length - {}'.format(
                            self.window_length))
                        self.test_feats = None
                        self.train_feats = None
                    else:
                        window_length_list = trajectory_windows_list

                    self.train_feats, self.test_feats = self.generate_topological_features(X_train=self.X_train,
                                                                                           X_test=self.X_test,
                                                                                           tokens_embedding_delay=self.window_length)
                    predictor = self.fit(X_train=self.train_feats,
                                         y_train=self.y_train,
                                         window_length=self.window_length)

                    result_on_train = self._predict_on_train(predictor=predictor)

                    self.X_test = self.test_feats

                    self._get_dimension_params(predictions_proba_train=result_on_train['predictions_proba'])

                    result_on_test = self._predict_on_test(predictor=predictor)

                    if save_features:
                        pd.DataFrame(self.train_feats).to_csv(os.path.join(self.path_to_save, 'train_features.csv'))
                        pd.DataFrame(self.y_train).to_csv(os.path.join(self.path_to_save, 'train_target.csv'))
                        pd.DataFrame(self.test_feats).to_csv(os.path.join(self.path_to_save, 'test_features.csv'))
                        pd.DataFrame(self.y_test).to_csv(os.path.join(self.path_to_save, 'test_target.csv'))

                    self._save_all_results(predictor=predictor,
                                           boosting_results=None,
                                           normal_results=result_on_test)

                    # boosting_results = self._predict_with_boosting(predictions=result_on_test['predictions'],
                    #                                                predictions_proba=result_on_test[
                    #                                                    'predictions_proba'],
                    #                                                metrics_without_boosting=result_on_test['metrics'])
                    # boosting_results['dataset'] = dataset
                    #
                    # self._save_all_results(predictor=predictor,
                    #                        boosting_results=boosting_results,
                    #                        normal_results=result_on_test,
                    #                        save_boosting=True)

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
