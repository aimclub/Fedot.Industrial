from sklearn.ensemble import IsolationForest as SklearnIsolationForest
import numpy as np


class IsolationForest:
    """
    Isolation Forest or iForest builds an ensemble of iTrees for a given data set, then anomalies are those
    instances which have short average path lengths on the iTrees.

    Parameters
    ----------
    params : list
        A list containing three parameters: random_state, n_jobs, and contamination.
        
    Attributes
    ----------
    random_state : int
        The random seed used for reproducibility.
    n_jobs : int
        The number of CPU cores to use for parallelism.
    contamination : float
        The expected proportion of anomalies in the dataset.
    """

    def __init__(self, params):
        self.model = None
        self.params = params
        self.random_state = self.params[0]
        self.n_jobs = self.params[1]
        self.contamination = self.params[2]

    def _build_model(self):
        model = SklearnIsolationForest(random_state=self.random_state,
                                       n_jobs=self.n_jobs,
                                       contamination=self.contamination)
        return model

    def fit(self, input_array: np.array):
        """
        Train the Isolation Forest model on the provided data.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for training the model.
        """
        self.model = self._build_model()
        self.model.fit(input_array)

    def predict(self, input_array: np.array):
        """
        Generate predictions using the trained Isolation Forest model.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        return self.model.predict(input_array)
