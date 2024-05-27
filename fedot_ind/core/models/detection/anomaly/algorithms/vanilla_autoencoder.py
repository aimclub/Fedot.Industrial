from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class VanillaAutoEncoder:
    """
    Feed-forward neural network with autoencoder architecture for anomaly detection using reconstruction error
    as an anomaly score.

    Parameters
    ----------
    params : list
        List containing the following hyperparameters in order:
            - Number of neurons in the first encoder layer
            - Number of neurons in the bottleneck layer (latent representation)
            - Number of neurons in the first decoder layer
            - Learning rate for the optimizer
            - Batch size for training

    Attributes
    ----------
    model : tensorflow.keras.models.Model
        The autoencoder model.
    """

    def __init__(self, params):
        self.model = None
        self.shape = None
        self.param = params

    def _build_model(self):
        input_dots = Input(shape=(self.shape,))
        x = Dense(self.param[0])(input_dots)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(self.param[1])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        bottleneck = Dense(self.param[2], activation='linear')(x)
        x = Dense(self.param[1])(bottleneck)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(self.param[0])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        out = Dense(self.shape, activation='linear')(x)
        model = Model(input_dots, out)
        model.compile(
            optimizer=Adam(
                self.param[3]),
            loss='mae',
            metrics=["mse"])
        self.model = model

        return model

    def fit(
            self,
            data,
            early_stopping=True,
            validation_split=0.2,
            epochs=40,
            verbose=0,
            shuffle=True):
        """
        Train the autoencoder model on the provided data.

        Parameters
        ----------
        data : np.ndarray
            Input data for training.
        early_stopping : bool, optional
            Whether to use early stopping during training.
        validation_split : float, optional
            Fraction of the training data to be used as validation data.
        epochs : int, optional
            Number of training epochs.
        verbose : int, optional
            Verbosity mode (0 = silent, 1 = progress bar, 2 = current epoch and losses, 3 = each training iteration).
        shuffle : bool, optional
            Whether to shuffle the training data before each epoch.
        """
        self.shape = data.shape[1]
        self.model = self._build_model()
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(patience=3, verbose=0))
        self.model.fit(data, data,
                       validation_split=validation_split,
                       epochs=epochs,
                       batch_size=self.param[4],
                       verbose=verbose,
                       shuffle=shuffle,
                       callbacks=callbacks
                       )

    def predict(self, data):
        """
        Generate predictions using the trained autoencoder model.

        Parameters
        ----------
        data : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            The reconstructed output predictions.
        """
        return self.model.predict(data)
