from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


class LSTMAutoEncoder:
    """
    A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data
    using reconstruction error as an anomaly score.

    Parameters
    ----------
    params : list
        A list of hyperparameters for the model, containing the following elements:
        - EPOCHS : int
            The number of training epochs.
        - BATCH_SIZE : int
            The batch size for training.
        - VAL_SPLIT : float
            The validation split ratio during training.

    Attributes
    ----------
    params : list
        The hyperparameters for the model.
    """

    def __init__(self, params):
        self.shape = None
        self.model = None
        self.params = params

    def _build_model(self):
        inputs = Input(shape=(self.shape[1], self.shape[2]))
        encoded = LSTM(100, activation='relu')(inputs)

        decoded = RepeatVector(self.shape[1])(encoded)
        decoded = LSTM(100, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.shape[2]))(decoded)

        model = Model(inputs, decoded)
        # TODO: not used
        Model(inputs, encoded)

        model.compile(optimizer='adam', loss='mae', metrics=["mse"])

        return model

    def fit(self, input_array: np.array):
        """
        Train the sequence-to-sequence (LSTM-based) autoencoder model on the provided data.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for training the model.
        """

        self.shape = input_array.shape
        self.model = self._build_model()

        early_stopping = EarlyStopping(patience=5,
                                       verbose=0)

        self.model.fit(
            input_array,
            input_array,
            validation_split=self.params[2],
            epochs=self.params[0],
            batch_size=self.params[1],
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping]
        )

    def predict(self, input_array: np.array):
        """
        Generate predictions using the trained sequence-to-sequence (LSTM-based) autoencoder model.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for generating predictions.

        Returns
        -------
        np.ndarray
            Predicted output data.
        """
        return self.model.predict(input_array)
