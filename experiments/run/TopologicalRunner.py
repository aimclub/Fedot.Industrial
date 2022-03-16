import json
from multiprocessing.dummy import Pool
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.model_selection import train_test_split
from core.topological.external.TFE import *
from core.topological.external import *
from core.topological.TDA import Topological
from experiments.run.ExperimentRunner import ExperimentRunner
from utils.utils import *
import timeit


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
            persistence_diagram_features=[HolesNumberFeature(),
                                          MaxHoleLifeTimeFeature(),
                                          RelevantHolesNumber(),
                                          AverageHoleLifetimeFeature(),
                                          SumHoleLifetimeFeature(),
                                          PersistenceEntropyFeature(),
                                          SimultaneousAliveHolesFeatue(),
                                          AveragePersistenceLandscapeFeature(),
                                          BettiNumbersSumFeature(),
                                          RadiusAtMaxBNFeature()])

        X_train_transformed = feature_extractor.fit_transform(X_train.values)
        X_test_transformed = feature_extractor.fit_transform(X_test.values)
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

    def _create_path_to_save(self, dataset, launch):
        save_path = os.path.join(path_to_save_results(), dataset, str(launch))
        return save_path

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length: int = None):

        self.logger.info('Generating features for fit model')
        self.window_length = window_length
        # train_feats, betti_df_train = self.generate_features_from_ts(X_train)

        self.logger.info('Start fitting FEDOT model')

        converted_features = self.__convert_to_input_data(features=X_train,
                                                          target=y_train,
                                                          col_numbers=X_train.shape[1])

        predictor = Fedot(**self.fedot_params)
        predictor.fit(converted_features)
        return predictor

    def predict(self, predictor,
                X_test: pd.DataFrame,
                window_length: int = None,
                y_test: np.array = None):
        self.logger.info('Generating features for prediction')

        #test_feats, betti_df_test = self.generate_features_from_ts(ts_frame=X_test, window_length=window_length)

        converted_features = self.__convert_to_input_data(features=X_test,
                                                          target=y_test,
                                                          col_numbers=X_test.shape[1])

        start_time = timeit.default_timer()
        predictions = predictor.predict(features=converted_features)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=converted_features)

        return predictions, predictions_proba, inference

    def run_experiment(self,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict):
        for dataset in self.list_of_dataset:
            for launch in range(self.launches):
                try:
                    path_to_save = self._create_path_to_save(dataset, launch)
                    X, y = dict_of_dataset[dataset]

                    if type(X) is tuple:
                        X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))

                    X_train_converted, X_test_converted = self.generate_topological_features(X_train,
                                                                                             X_test,
                                                                                             dict_of_win_list[dataset])
                    predictor = self.fit(X_train=X_train_converted,
                                         y_train=y_train,
                                         window_length=dict_of_win_list[dataset])

                    self.count = 0

                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test_converted,
                                                                             window_length=dict_of_win_list[dataset],
                                                                             y_test=y_test)

                    self.logger.info('Saving model')
                    predictor.current_pipeline.save(path=path_to_save)
                    best_pipeline, fitted_operation = predictor.current_pipeline.save()

                    try:
                        opt_history = predictor.history.save()
                        with open(os.path.join(path_to_save, 'history', 'opt_history.json'), 'w') as f:
                            json.dump(json.loads(opt_history), f)
                    except Exception as ex:
                        ex = 1

                    self.logger.info('Saving results')
                    try:
                        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                                  target=y_test,
                                                                  predicted_labels=predictions,
                                                                  predicted_probs=predictions_proba)
                    except Exception as ex:
                        metrics = 'empty'

                    save_results(predictions=predictions,
                                 prediction_proba=predictions_proba,
                                 target=y_test,
                                 metrics=metrics,
                                 inference=inference,
                                 fit_time=np.mean(self._generate_fit_time(predictor)),
                                 path_to_save=path_to_save)
                    self.count = 0

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
