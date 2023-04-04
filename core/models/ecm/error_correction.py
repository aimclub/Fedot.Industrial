import logging
import warnings

import numpy as np
import pandas as pd
from fedot.api.main import Fedot

warnings.simplefilter(action='ignore', category=FutureWarning)


class Booster:
    """Class to implement error correction model (ECM) for classification prediction improvement.

    Args:
        features_train: X_train
        target_train: y_train
        base_predict: prediction, derived from main model (Quantile, Spectral, Topological, or Wavelet)
        timeout: defines the amount of time to compose and tune regression model
        threshold: parameter used as round boundary for custom_round() method
        n_cycles: number of boosting cycles
        reshape_flag: ...

    Examples:
        >>> X_train = pd.read_csv('X_train.csv')
        >>> y_train = pd.read_csv('y_train.csv')
        >>> base_predict = pd.read_csv('base_predict.csv')

        >>> y_train = y_train.values
        >>> base_predict = base_predict.values

        >>> booster = Booster(features_train=X_train,
                              target_train=y_train,
                              base_predict=base_predict,
                              timeout=1)

        >>> result = booster.fit()
    """

    def __init__(self, features_train: np.ndarray,
                 target_train: np.array,
                 base_predict: np.array,
                 timeout: int,
                 threshold: float = 0.5,
                 reshape_flag: bool = False,
                 n_cycles: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.X_train = features_train
        self.target_train = target_train
        self.base_predict = base_predict
        self.threshold = threshold
        self.timeout = max(3, round(timeout / 4))
        # self.timeout = 1
        self.reshape_flag = reshape_flag
        self.n_cycles = n_cycles
        if self.reshape_flag:
            self.target_train = self.target_train.reshape(-1)

        self.ecm_targets = [self.target_train, ]
        self.ecm_predictions = [self.base_predict, ]

    def fit(self) -> tuple:
        """Method to fit error correction models.

        Returns:
            tuple: tuple with ensemble prediction and ensemble model
        """
        self.logger.info('Started boosting')
        model_list = list()

        for i in range(self.n_cycles):
            self.logger.info(f'Starting cycle {1 + i} of boosting')
            target_diff = np.subtract(previous_predict=self.ecm_predictions[i],
                                      previous_target=self.ecm_targets[i])
            self.ecm_targets.append(target_diff)

            prediction, fedot_model = self.api_model(target_diff=target_diff)
            self.ecm_predictions.append(prediction)
            model_list.append(fedot_model)

        final_prediction, model_ensemble = self.ensemble(features=self.ecm_predictions[1:])

        return final_prediction, model_list, model_ensemble

    def api_model(self, target_diff: np.ndarray) -> tuple:
        """Method used to initiate FEDOT AutoML model to solve regression problem for boosting stage

        Args:
            target_diff: target difference between previous prediction and target

        Returns:
            Tuple with prediction and FEDOT model
        """
        fedot_model = Fedot(problem='regression', seed=42,
                            timeout=self.timeout, max_depth=10,
                            max_arity=4,
                            cv_folds=2,
                            logging_level=20, n_jobs=2)

        fedot_model.fit(self.X_train, target_diff)
        prediction = fedot_model.predict(self.X_train)

        if self.reshape_flag:
            prediction = prediction.reshape(-1)

        return prediction, fedot_model

    def ensemble(self, features: list) -> tuple:
        """Method that ensembles results of all stages of boosting. Depending on number of classes ensemble method
        could be a genetic AutoML model by FEDOT (for binary problem) or SUM method (for multi-class problem).

        Args:
            features: list of predictions from all boosting stages

        Returns:
            Tuple with ensemble prediction and ensemble model

        """
        self.logger.info('Starting to ensemble boosting results')

        ensemble_model = Fedot(problem='regression',
                               timeout=self.timeout,
                               seed=20,
                               logging_level=20,
                               n_jobs=6)
        if self.reshape_flag:
            features = np.hstack(features)
            target = self.target_train

            ensemble_model.fit(features, target)
            ensemble_prediction = ensemble_model.predict(features)
            ensemble_prediction = ensemble_prediction.reshape(-1)
            return ensemble_prediction, ensemble_model

        else:
            ensemble_prediction = sum(features)
            return ensemble_prediction, None

    @staticmethod
    def proba_to_vector(matrix: np.array):
        """Method to convert probability matrix to vector of labels.

        Args:
            matrix: probability matrix

        Returns:
            Probability matrix received as a result of prediction_proba for multi-class problem

        """
        vector = np.array([x.argmax() + x[x.argmax()] for x in matrix])
        return vector

    def custom_round(self, num: float) -> int:
        """Custom round method with predefined threshold.

        Args:
            num: number to round

        Returns:
            Rounded number
        """
        thr = self.threshold
        if num - int(num) >= thr:
            return int(num) + 1
        return int(num)

