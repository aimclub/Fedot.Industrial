import numpy as np
from tensorflow.keras.layers import Input, Conv1D, Dropout, Conv1DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class ConvolutionalAutoEncoder:
    """
    A reconstruction convolutional autoencoder model to detect anomalies in timeseries data using reconstruction
    error as an anomaly score.

    Parameters
    ----------
    No parameters are required for initializing the class.

    Attributes
    ----------
    model : Sequential
        The trained convolutional autoencoder model.
    """
    
    def __init__(self):
        self.model = None
        self.shape = None

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.shape[1], self.shape[2])),
                Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model
    
    def fit(self, input_array: np.array):
        """
        Train the convolutional autoencoder model on the provided data.

        Parameters
        ----------
        input_array : np.ndarray
            Input data for training the autoencoder model.
        """
        
        self.shape = input_array.shape
        self.model = self._build_model()

        self.model.fit(
            input_array,
            input_array,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )

    def predict(self, input_array: np.array):
        """
        Generate predictions using the trained convolutional autoencoder model.

        Parameters
        ----------
        input_array : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(input_array)
