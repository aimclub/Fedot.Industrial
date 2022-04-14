import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from cases.analyzer import PerfomanceAnalyzer
from cases.run.utils import get_logger
from fedot.api.main import Fedot

import numpy as np
import timeit


class Booster:
    CYCLES = 0

    def __init__(self,
                 previous_predict,
                 input_data: tuple = None,
                 metrics_name: list = ('f1',
                                       'roc_auc',
                                       'accuracy',
                                       'logloss',
                                       'precision'),
                 ):
        self.previous_predict = previous_predict.reshape(previous_predict.shape[0],)

        self.boost_predictions = pd.DataFrame({'0': self.previous_predict})

        self.features = input_data[0]
        self.target = input_data[1]
        self.metrics_name = metrics_name

        self.accuracy_base = accuracy_score(self.target, self.previous_predict)
        self.f1_base = f1_score(self.target, self.previous_predict, average='weighted')

        self.logger = get_logger()

    def get_sum(self):
        columns = self.boost_predictions.columns
        self.boost_predictions.columns['sum'] = 0
        for col in columns:
            self.boost_predictions.columns['sum'] += self.boost_predictions.columns[col]

        return self.boost_predictions.columns['sum'].values

    def run_boosting(self):
        last_predict = self.previous_predict
        self.logger.info('start cycle')
        predictions, inference = self.run_cycle(last_predict)
        predictions, inference = self.run_cycle(last_predict)

        return predictions, inference

    def run_cycle(self, last_predict):
        self.CYCLES += 1
        self.logger.info('-------calculate diff')
        diff = self.target - last_predict

        predictor = Fedot(problem='regression',
                          timeout=2,
                          seed=20,
                          verbose_level=1,
                          n_jobs=-1)
        X, y = self.features, self.target

        self.logger.info('-------splitting set')
        X_train, _, y_train, _ = train_test_split(X,
                                                  diff,
                                                  test_size=0.2,
                                                  random_state=np.random.randint(100))
        self.logger.info('-------fitting model')

        predictor.fit(X_train, y_train)
        self.logger.info('-------prediction model')
        start_time = timeit.default_timer()
        predictions = predictor.predict(X)
        inference = timeit.default_timer() - start_time
        # predictions_proba = predictor.predict_proba(self.features)
        self.logger.info('-------filling dataframe')
        self.boost_predictions[self.CYCLES] = predictions
        self.logger.info('-------getting sum of predictions')
        result_prediction = self.get_sum()

        self.logger.info('-------calc accuracy model')
        accuracy_before_boosting = accuracy_score(y, self.previous_predict)
        accuracy_after_boosting = accuracy_score(y, result_prediction)
        acc_boost = 100 * accuracy_after_boosting / accuracy_before_boosting
        self.logger.info('-------calc F1 metric model')
        f1_before_boosting = f1_score(y, self.previous_predict, average='weighted')
        f1_after_boosting = f1_score(y, result_prediction, average='weighted')
        f1_boost = 100 * f1_after_boosting / f1_before_boosting

        self.logger.info(f'{self.CYCLES} cycles of boosting increased accuracy to {acc_boost}%')
        self.logger.info(f'{self.CYCLES} cycles of boosting increased F1 to {f1_boost}%')

        return predictions, inference
