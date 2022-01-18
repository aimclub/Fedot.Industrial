from fedot.api.main import Fedot
from sklearn.model_selection import train_test_split
from sktime.datasets import *
from experiments.analyzer import PerfomanceAnalyzer
from core.spectral.SSA import Spectrum
from core.statistical.Stat_features import AggregationFeatures
from utils.utils import *
import timeit



dict_of_dataset = {'gunpoint': load_gunpoint(return_X_y=True),
                   'basic_motions': load_basic_motions(return_X_y=True),
                   'arrow_head': load_arrow_head(return_X_y=True),
                   'osuleaf': load_osuleaf(return_X_y=True),
                   'italy_power': load_italy_power_demand(return_X_y=True),
                   'unit_test': load_unit_test(return_X_y=True)
                   }

dict_of_win_list = {'gunpoint': 30,
                    'basic_motions': 10,
                    'arrow_head': 50,
                    'osuleaf': 90,
                    'italy_power': 3,
                    'unit_test': 3
                    }


class ExperimentRunner:
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 1,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']):
        self.aggregator = AggregationFeatures()
        self.spectrum_extractor = Spectrum
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name

    def generate_features_from_ts(self, ts_frame, window_length=None):
        new_features = []
        for ts in ts_frame['dim_0'].values:
            spectr = self.spectrum_extractor(time_series=ts,
                                             window_length=window_length)
            TS_comps, X_elem, V, Components_df, _ = spectr.decompose()
            aggregation_df = self.aggregator.create_features(feature_to_aggregation=Components_df.iloc[:, :5])
            new_features.append(aggregation_df)
            feats = pd.concat(new_features)
        return feats

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

        train_feats = self.generate_features_from_ts(X_train, window_length=window_length)
        predictor = Fedot(problem='classification', seed=42, timeout=2)
        predictor.fit(features=train_feats, target=y_train)
        return predictor

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None):
        test_feats = self.generate_features_from_ts(ts_frame=X_test, window_length=window_length)
        start_time = timeit.default_timer()
        predictions = predictor.predict(features=test_feats)
        inference = timeit.default_timer() - start_time
        predictions_proba = predictor.predict_proba(features=test_feats)
        return predictions, predictions_proba, inference

    def run_experiment(self):
        prediction_list = []
        target = []
        models = []
        for dataset in self.list_of_dataset:
            for launch in range(self.launches):
                try:
                    path_to_save = self._create_path_to_save(dataset, launch)
                    X, y = dict_of_dataset[dataset]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))
                    predictor = self.fit(X_train=X_train,
                                         y_train=y_train,
                                         window_length=dict_of_win_list[dataset])
                    predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                             X_test=X_test,
                                                                             window_length=dict_of_win_list[dataset])
                    _ =1
                    try:
                        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                                  target=target,
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

                except Exception as ex:
                    print(ex)
                    print(str(dataset))
        return prediction_list, target, models


if __name__ == '__main__':
    list_of_dataset = ['italy_power']
    runner = ExperimentRunner(list_of_dataset)
    preds, target, models = runner.run_experiment()
    _ = 1
