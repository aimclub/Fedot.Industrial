import json
import os.path
from core.metrics.metrics_implementation import *
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from core.operation.utils.booster import Booster
from core.operation.utils.static_booster import StaticBooster
from sklearn.model_selection import train_test_split
from cases.analyzer import PerfomanceAnalyzer
from core.operation.utils.utils import *
from cases.run.utils import *

dict_of_dataset = dict
dict_of_win_list = dict


class ExperimentRunner:
    def __init__(self,
                 list_of_dataset: list = None,
                 launches: int = 3,
                 metrics_name: list = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision'],
                 fedot_params: dict = {'problem': 'classification',
                                       'seed': 42,
                                       'timeout': 1,
                                       'composer_params': {'max_depth': 10,
                                                           'max_arity': 4},
                                       'verbose_level': 1},
                 boost_mode: bool = True,
                 static_booster: bool = False):
        self.analyzer = PerfomanceAnalyzer()
        self.list_of_dataset = list_of_dataset
        self.launches = launches
        self.metrics_name = metrics_name
        self.count = 0
        self.window_length = None
        self.logger = get_logger()
        self.fedot_params = fedot_params
        self.boost_mode = boost_mode

        self.static_booster = static_booster
        self.y_test = None

    def generate_features_from_ts(self, ts_frame, window_length=None):
        """  Method responsible for  experiment pipeline """
        return

    def _generate_fit_time(self, predictor):
        """  Method responsible for  experiment pipeline """
        return

    def _create_path_to_save(self, dataset, launch):
        """  Method responsible for  experiment pipeline """
        return

    def _load_data(self, dataset, dict_of_dataset):
        X, y = dict_of_dataset[dataset]
        if type(X) is tuple:
            X_train, X_test, y_train, y_test = X[0], X[1], y[0], y[1]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.randint(100))
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, window_length_list: list = None):
        """  Method responsible for  experiment pipeline """
        return

    def extract_features(self,
                         dataset_name: str,
                         dict_of_dataset: dict = None,
                         dict_of_extra_params: dict = None):
        return

    def predict(self, predictor, X_test: pd.DataFrame, window_length: int = None, y_test=None):
        """  Method responsible for  experiment pipeline """
        return

    def _get_clf_params(self):
        self.n_classes = np.unique(self.y_train)

        if self.n_classes.shape[0] > 2:
            self.fedot_params['composer_params']['metric'] = 'f1'
        else:
            self.fedot_params['composer_params']['metric'] = 'roc_auc'

    def _get_dimension_params(self, predictions_proba_train):
        if self.n_classes.shape[0] > 2:
            self.base_predict = predictions_proba_train
            y_train_multi = pd.get_dummies(self.y_train, sparse=True)
            self.y_train = y_train_multi.values
            self.reshape_flag = False
        else:
            self.base_predict = predictions_proba_train.reshape(-1)
            self.reshape_flag = True

    def _convert_boosting_prediction(self, boosting_test, ensemble_model, predictions_proba):
        if self.reshape_flag:
            boosting_test = [x.reshape(-1) for x in boosting_test]
            error_correction = ensemble_model.predict(pd.DataFrame(data=boosting_test).T)
            corrected_probs = error_correction.reshape(-1) + predictions_proba.reshape(-1)
            corrected_probs = corrected_probs.reshape(-1)
            corrected_labels = abs(np.round(corrected_probs))

        else:
            # error_correction = ensemble_model.predict(np.hstack(boosting_test))
            dictlist = []
            for value in boosting_test:
                dictlist.append(value)
            error_correction = sum(dictlist)
            corrected_probs = error_correction + predictions_proba
            corrected_labels = np.array([x.argmax() + min(self.y_test) for x in corrected_probs])
            corrected_probs = pd.get_dummies(corrected_labels, sparse=True).values
            corrected_labels = corrected_labels.reshape(len(corrected_labels), 1)

        return dict(corrected_labels=corrected_labels,
                    corrected_probs=corrected_probs,
                    error_correction=error_correction)

    def _validate_window_length(self, features, target):

        node = PrimaryNode('rf')
        pipeline = Pipeline(node)
        n_samples = round(features.shape[0] * 0.7)

        train_data = InputData(features=features.values[:n_samples, :], target=target[:n_samples],
                               idx=np.arange(0, len(target[:n_samples])),
                               task=Task(TaskTypesEnum('classification')), data_type=DataTypesEnum.table)
        test_data = InputData(features=features.values[n_samples:, :], target=target[n_samples:],
                              idx=np.arange(0, len(target[n_samples:])),
                              task=Task(TaskTypesEnum('classification')), data_type=DataTypesEnum.table)

        fitted = pipeline.fit(input_data=train_data)
        prediction = pipeline.predict(input_data=test_data, output_mode='labels')
        metric_f1 = F1()
        metric_roc = ROCAUC()

        score_f1 = metric_f1.metric(target=prediction.target, prediction=prediction.predict)

        try:
            try:
                score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
            except Exception:
                prediction = pipeline.predict(input_data=test_data, output_mode='probs')
                score_roc_auc = metric_roc.metric(target=prediction.target, prediction=prediction.predict)
        except Exception:
            score_roc_auc = 0.5

        return score_f1, score_roc_auc

    def _predict_on_train(self, predictor):

        self.count = 0
        self.test_feats = self.train_feats

        # Predict on whole TRAIN
        predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                 X_test=self.X_train,
                                                                 window_length=self.window_length,
                                                                 y_test=self.y_train)

        # GEt metrics on whole TRAIN

        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                  target=self.y_train,
                                                  predicted_labels=predictions,
                                                  predicted_probs=predictions_proba
                                                  )

        self.logger.info(f'Without Boosting metrics are: {metrics}')

        return dict(predictions=predictions,
                    predictions_proba=predictions_proba,
                    inference=inference,
                    metrics=metrics)

    def _predict_on_test(self, predictor):

        self.test_feats = None
        predictions, predictions_proba, inference = self.predict(predictor=predictor,
                                                                 X_test=self.X_test,
                                                                 window_length=self.window_length,
                                                                 y_test=self.y_test)

        metrics = self.analyzer.calculate_metrics(self.metrics_name,
                                                  target=self.y_test,
                                                  predicted_labels=predictions,
                                                  predicted_probs=predictions_proba
                                                  )

        return dict(predictions=predictions,
                    predictions_proba=predictions_proba,
                    inference=inference,
                    metrics=metrics)

    def static_boosting_pipeline(self, predictions_proba, model_list, ensemble_model):
        boosting_test = []
        task = Task(TaskTypesEnum.regression)
        input_data_test = InputData(idx=np.arange(0, len(self.test_feats)),
                                    features=self.test_feats,
                                    target=self.y_test,
                                    task=task,
                                    data_type=DataTypesEnum.table)

        for model in model_list:
            boost_predict = model.predict(input_data=input_data_test).predict.reshape(-1)
            boosting_test.append(boost_predict)

        boosting_test = pd.DataFrame([x.reshape(-1) for x in boosting_test]).T

        input_data = InputData(idx=np.arange(0, len(boosting_test)),
                               features=boosting_test,
                               target=self.y_test,
                               task=task,
                               data_type=DataTypesEnum.table)
        error_correction = ensemble_model.predict(input_data=input_data).predict.reshape(-1)
        boosting_result = self._convert_boosting_prediction(boosting_test=boosting_test,
                                                            ensemble_model=ensemble_model,
                                                            predictions_proba=predictions_proba)
        return boosting_result

    def genetic_boosting_pipeline(self, predictions_proba, model_list, ensemble_model):

        pass

    def _predict_witn_boosting(self,
                               predictions,
                               predictions_proba,
                               metrics_without_boosting):

        if self.static_booster:
            booster = StaticBooster(X_train=self.train_feats,
                                    y_train=self.y_train,
                                    base_predict=self.base_predict,
                                    timeout=round(self.fedot_params['timeout'] / 2),
                                    )
            boosting_pipeline = self.static_boosting_pipeline
        else:
            booster = Booster(X_train=self.train_feats,
                              y_train=self.y_train,
                              base_predict=self.base_predict,
                              timeout=round(self.fedot_params['timeout'] / 2),
                              )
            boosting_pipeline = self.genetic_boosting_pipeline

        predictions_boosting_train, model_list, ensemble_model = booster.run_boosting()
        results = boosting_pipeline(predictions_proba, model_list, ensemble_model)
        results['base_probs'] = predictions_proba
        results['true_labels'] = predictions

        solution_table = pd.DataFrame({'target': self.y_test,
                                       'base_probs': results['base_probs'].reshape(-1),
                                       'ensemble': results['ensemble'].reshape(-1),
                                       'corrected_probs': results['corrected_probs'],
                                       'corrected_labels': results['corrected_labels'],
                                       'true_labels': results['true_labels']})

        boosting_test = []
        for model in model_list:
            if self.static_booster:
                task = Task(TaskTypesEnum.regression)
                input_data_test = InputData(idx=np.arange(0, len(self.test_feats)),
                                            features=self.test_feats,
                                            target=self.y_test,
                                            task=task,
                                            data_type=DataTypesEnum.table)
                boost_predict = model.predict(input_data=input_data_test).predict.reshape(-1)
                boosting_test.append(boost_predict)
            else:
                boosting_test.append(model.predict(input_data=self.test_feats))

        boosting_test = pd.DataFrame([x.reshape(-1) for x in boosting_test]).T

        if self.static_booster:
            task = Task(TaskTypesEnum.regression)
            input_data = InputData(idx=np.arange(0, len(boosting_test)),
                                   features=boosting_test,
                                   target=self.y_test,
                                   task=task,
                                   data_type=DataTypesEnum.table)
            error_correction = ensemble_model.predict(input_data=input_data).predict.reshape(-1)
        else:
            error_correction = ensemble_model.predict(boosting_test).predict.reshape(-1)

        corrected_probs = error_correction + predictions_proba.reshape(-1)
        corrected_probs = corrected_probs.reshape(-1)
        corrected_labels = abs(np.round(corrected_probs))

        for index in boosting_test.columns:
            solution_table[f'boost_{index + 1}'] = boosting_test[index]

            boosting_test.append(model.predict(self.test_feats))

        metrics_boosting = self.analyzer.calculate_metrics(self.metrics_name,
                                                           target=self.y_test,
                                                           predicted_labels=corrected_labels,
                                                           predicted_probs=corrected_probs
                                                           )

        no_boost = pd.Series(metrics_without_boosting)
        boost = pd.Series(metrics_boosting)
        metrics_table = pd.concat([no_boost, boost], axis=1).reset_index()
        metrics_table.columns = ['metric', 'before_boosting', 'after_boosting']

        return dict(solution_table=solution_table,
                    metrics_table=metrics_table,
                    model_list=model_list,
                    ensemble_model=ensemble_model)

    def _save_all_results(self, predictor, boosting_results, normal_results):

        self.logger.info('Saving model')
        predictor.current_pipeline.save(path=self.path_to_save)
        best_pipeline, fitted_operation = predictor.current_pipeline.save()

        try:
            opt_history = predictor.history.save()
            with open(os.path.join(self.path_to_save, 'history', 'opt_history.json'), 'w') as f:
                json.dump(json.loads(opt_history), f)
        except Exception as ex:
            ex = 1

        self.logger.info('Saving results')
        self.save_boosting_results(**boosting_results)
        save_results(**normal_results)

    def run_experiment(self,
                       dict_of_dataset: dict,
                       dict_of_win_list: dict):

        for dataset in self.list_of_dataset:

            self.train_feats = None
            self.test_feats = None

            for launch in range(self.launches):
                try:
                    self.path_to_save = self._create_path_to_save(dataset, launch)
                    self.path_to_save_png = os.path.join(self.path_to_save, 'pictures')

                    if not os.path.exists(self.path_to_save_png):
                        os.makedirs(self.path_to_save_png)

                    X, y = dict_of_dataset[dataset]
                    self.X_train, self.X_test, self.y_train, self.y_test = X[0], X[1], y[0], y[1]

                    self._get_clf_params()

                    predictor = self.fit(X_train=self.X_train,
                                         y_train=self.y_train,
                                         window_length_list=dict_of_win_list[dataset])

                    result_on_train = self._predict_on_train(predictor=predictor)

                    self._get_dimension_params(predictions_proba_train=result_on_train['prediction_proba'])

                    result_on_test = self._predict_on_test(predictor=predictor)

                    result_with_boosting = self._predict_witn_boosting(predictions=result_on_train['predictions'],
                                                                       predictions_proba=result_on_train[
                                                                           'predictions_proba'],
                                                                       metrics_without_boosting=result_on_test[
                                                                           'metrics'])
                    result_with_boosting['dataset'] = dataset

                    self._save_all_results(predictor=predictor,
                                           boosting_results=result_with_boosting,
                                           normal_results=result_on_test)
                except Exception:
                    print('Problem')

    @staticmethod
    def save_boosting_results(dataset, solution_table, metrics_table, model_list, ensemble_model):
        location = os.path.join(project_path(), 'results_of_experiments', dataset, 'boosting')
        if not os.path.exists(location):
            os.makedirs(location)
        solution_table.to_csv(os.path.join(location, 'solution_table.csv'))
        metrics_table.to_csv(os.path.join(location, 'metrics_table.csv'))

        models_path = os.path.join(location, 'boosting_pipelines')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        for index, model in enumerate(model_list):
            try:
                model.current_pipeline.save(path=os.path.join(models_path, f'boost_{index}'),
                                            datetime_in_path=False)
                ensemble_model.current_pipeline.save(path=os.path.join(models_path, 'boost_ensemble'),
                                                     datetime_in_path=False)
            except Exception:
                model.save(path=os.path.join(models_path, f'boost_{index}'),
                           datetime_in_path=False)
                ensemble_model.save(path=os.path.join(models_path, 'boost_ensemble'),
                                    datetime_in_path=False)
