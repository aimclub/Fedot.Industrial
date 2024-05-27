import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class VanillaLSTM:
    """
    LSTM-based neural network for anomaly detection using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list containing various parameters for configuring the LSTM model.

    Attributes
    ----------
    model : Sequential
        The trained LSTM model.
    """

    def __init__(self, params):
        self.model = None
        self.n_features = None
        self.params = params

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(100,
                       activation='relu',
                       return_sequences=True,
                       input_shape=(self.params[0], self.n_features)))
        model.add(LSTM(100,
                       activation='relu'))
        model.add(Dense(self.n_features))
        model.compile(optimizer='adam',
                      loss='mae',
                      metrics=["mse"])
        return model

    def fit(self, input_array: np.array, y: np.array):
        """
        Train the LSTM model on the provided data.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for training the model.
        y : np.ndarray
            Target data for training the model.
        """
        self.n_features = input_array.shape[2]
        self.model = self._build_model()
        early_stopping = EarlyStopping(patience=10, verbose=0)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1, patience=5, min_lr=0.0001, verbose=0)
        self.model.fit(
            input_array,
            y,
            validation_split=self.params[3],
            epochs=self.params[1],
            batch_size=self.params[2],
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping, reduce_lr]
        )

    def predict(self, data: np.array):
        """
        Generate predictions using the trained LSTM model.

        Parameters
        ----------
        data : np.ndarray
            Input data for generating predictions.

        Returns
        -------
        np.ndarray
            Predicted output data.
        """
        return self.model.predict(data)
