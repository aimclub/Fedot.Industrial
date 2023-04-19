import numpy as np
import pandas as pd

import logging
from fedot_ind.core.architecture.postprocessing.Analyzer import PerformanceAnalyzer
from fedot_ind.core.models.ecm.error_correction import Booster


class ErrorCorrectionModel:
    """Error correction model (ECM) for time series classification predictions.
    It is based on the idea of boosting: the model is trained on the residuals of the previous model predictions
    and the final prediction is the sum of the predictions of all models or another Fedot AutoML pipeline.

    Args:
        results_on_train: np.ndarray with proba_predictions on train data
        results_on_test: dictionary with predictions, proba_predictions and metrics on test data
        n_classes: number of classes
        dataset_name: name of dataset
        save_models: flag for saving ECM models
        fedot_params: dictionary with parameters for Fedot AutoML model
        train_data: train features and target
        test_data: test features and target

    """

    def __init__(self, results_on_train: np.ndarray = None,
                 results_on_test: dict = None, n_classes=None,
                 dataset_name: str = None, save_models: bool = False,
                 fedot_params: dict = None, train_data=None, test_data=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.results_on_train = results_on_train
        self.results_on_test = results_on_test
        self.n_classes = n_classes
        self.dataset_name = dataset_name
        self.save_models = save_models
        self.fedot_params = fedot_params
        self.train_feats, self.test_features = train_data[0], test_data[0]
        self.y_train, self.y_test = train_data[1], test_data[1]
        self.analyzer = PerformanceAnalyzer()

    def run(self):
        self.logger.info('Start error correction')
        self._get_dimension_params(predictions_proba_train=self.results_on_train)

        boosting_results = self._predict_with_boosting(predictions_proba=self.results_on_test['predictions_proba'],
                                                       metrics_without_boosting=self.results_on_test['metrics'])

        return boosting_results

    def _get_dimension_params(self, predictions_proba_train: np.ndarray) -> None:
        """Creates base predict for boosting and reshape flag for predictions.

        Args:
            predictions_proba_train: array with proba_predictions on train data

        """
        if self.n_classes > 2:
            self.base_predict = predictions_proba_train
            y_train_multi = pd.get_dummies(self.y_train, sparse=True)
            self.y_train = y_train_multi.values
            self.reshape_flag = False
        else:
            self.base_predict = predictions_proba_train.reshape(-1)
            self.base_predict = round(np.max(self.y_test) - np.max(predictions_proba_train)) + self.base_predict
            self.reshape_flag = True

    def _predict_with_boosting(self, predictions_proba: np.ndarray,
                               metrics_without_boosting: dict) -> dict:
        """Instantiates the Booster class and predicts on test features using boosting pipeline consists of
        several models and ensemble method.

        Args:
            predictions_proba: np.ndarray with proba_predictions on test data
            metrics_without_boosting: dictionary with metrics on test data without boosting

        Returns:
            boosting_results (dict): dictionary with predictions, proba_predictions
            and metrics on test data with boosting

        """
        booster = Booster(features_train=self.train_feats,
                          target_train=self.y_train,
                          base_predict=self.base_predict,
                          timeout=1,
                          reshape_flag=self.reshape_flag)

        boosting_pipeline = self.genetic_boosting_pipeline

        _, model_list, ensemble_model = booster.fit()
        results_on_test = boosting_pipeline(self.results_on_test['prediction'],
                                            model_list,
                                            ensemble_model)

        results_on_test['base_probs_on_test'] = predictions_proba

        proba_to_vector = booster.proba_to_vector

        solution_table = pd.DataFrame({'target': self.y_test,
                                       'base_probs_on_test': proba_to_vector(results_on_test['base_probs_on_test']),
                                       'corrected_probs': proba_to_vector(results_on_test['corrected_probs']),
                                       'corrected_labels': proba_to_vector(results_on_test['corrected_labels'])})

        self.metrics_name = ['f1', 'roc_auc', 'accuracy', 'logloss', 'precision']
        metrics_boosting = self.analyzer.calculate_metrics(metric_list=self.metrics_name,
                                                           target=self.y_test,
                                                           predicted_labels=results_on_test['corrected_labels'],
                                                           predicted_probs=results_on_test['corrected_probs'])

        no_boost = pd.Series(metrics_without_boosting)
        boost = pd.Series(metrics_boosting)
        metrics_table = pd.concat([no_boost, boost], axis=1).reset_index()
        metrics_table.columns = ['metric', 'before_boosting', 'after_boosting']

        return dict(solution_table=solution_table,
                    metrics_table=metrics_table,
                    model_list=model_list,
                    ensemble_model=ensemble_model)

    def genetic_boosting_pipeline(self, predictions_proba, model_list, ensemble_model) -> dict:
        self.logger.info('Predict on booster models')
        boosting_stages_predict = []
        input_data_test = self.test_features

        n = 1
        for model in model_list:
            self.logger.info(f'Cycle {n} of boosting has started')
            boost_predict = model.predict(input_data_test)
            boosting_stages_predict.append(boost_predict)
            n += 1

        self.logger.info('Ensemble booster predictions')
        if ensemble_model:
            self.logger.info('Ensemble using FEDOT has been chosen')
            boosting_stages_predict = pd.DataFrame(i.reshape(-1) for i in boosting_stages_predict).T
        else:
            boosting_stages_predict = [np.array(_) for _ in boosting_stages_predict]
            self.logger.info('Ensemble using SUM method has been chosen')

        boosting_result = self._convert_boosting_prediction(boosting_stages_predict=boosting_stages_predict,
                                                            ensemble_model=ensemble_model,
                                                            predictions_proba=predictions_proba)
        return boosting_result

    def _convert_boosting_prediction(self, boosting_stages_predict, ensemble_model, predictions_proba):
        self.logger.info('Calculation of error correction by boosting')
        # if self.static_booster:
        #     task = Task(TaskTypesEnum.regression)
        #     input_data = InputData(idx=np.arange(0, len(boosting_stages_predict)),
        #                            features=boosting_stages_predict,
        #                            target=self.y_test,
        #                            task=task,
        #                            data_type=DataTypesEnum.table)
        # else:
        input_data = boosting_stages_predict

        if self.reshape_flag:
            error_correction = ensemble_model.predict(input_data).reshape(-1)
            corrected_probs = error_correction.reshape(-1) + predictions_proba.reshape(-1)

            if self.reshape_flag and np.min(self.y_test) == 1:
                corrected_probs = (error_correction.reshape(-1) + predictions_proba.reshape(-1)) / 2
            corrected_probs = corrected_probs.reshape(-1)
            corrected_labels = abs(np.round(corrected_probs))

        else:
            dictlist = []
            for value in boosting_stages_predict:
                dictlist.append(value)
            error_correction = sum(dictlist)
            corrected_probs = error_correction + predictions_proba
            corrected_labels = np.array([x.argmax() + min(self.y_test) for x in corrected_probs])
            corrected_labels = corrected_labels.reshape(len(corrected_labels), 1)

        return dict(corrected_labels=corrected_labels,
                    corrected_probs=corrected_probs,
                    error_correction=error_correction)
