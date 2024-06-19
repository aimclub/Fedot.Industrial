from collections import OrderedDict
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from torch import cuda, device, no_grad, float32, from_numpy, Tensor
from torch.nn import LSTM, Module, MSELoss, Linear, Sequential

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly.algorithms.autoencoder_detector import AutoEncoderDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class LSTMAutoEncoderDetector(AutoEncoderDetector):
    """A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data
    using reconstruction error as an anomaly score.
    """

    def build_model(self):
        self.params.update(**{'n_steps': self.n_steps, 'learning_rate': self.learning_rate})
        return LSTMAutoEncoder(self.params).to(device)


class LSTMAutoEncoder(Module):
    def __init__(self, params: Optional[OperationParameters] = None):
        super(LSTMAutoEncoder, self).__init__()
        self.n_steps = params.get('n_steps', 10)
        self.embedding_dim = params.get('embedding_dim', 100)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.encoder_layers = params.get('num_encoder_layers', 2)
        self.decoder_layers = params.get('num_decoder_layers', 2)
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        encoder_layer_dict = OrderedDict()
        encoder_layer_dict.update({'lstm': LSTM(input_size=self.n_steps,
                                                hidden_size=2 * self.embedding_dim,
                                                num_layers=self.encoder_layers,
                                                batch_first=True)})
        self.encoder = Sequential(encoder_layer_dict)

    def _build_decoder(self):
        decoder_layer_dict = OrderedDict()
        decoder_layer_dict.update({'lstm': LSTM(input_size=2 * self.embedding_dim,
                                                hidden_size=self.embedding_dim,
                                                num_layers=self.decoder_layers,
                                                batch_first=True),
                                   'linear': Linear(self.embedding_dim, self.n_steps)})
        self.decoder = Sequential(decoder_layer_dict)

    def forward(self, x):
        # _, (x, _) = self.lstm1(x)
        # x = x.repeat(self.n_steps, 1, 1).transpose(0, 1)
        # x, _ = self.lstm1(x)
        # return self.output_layer(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, data, epochs: int = 100, batch_size: int = 32):
        dataset = TensorDataset(Tensor(data))
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        criterion = MSELoss()

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            for seq_true in train_loader:
                optimizer.zero_grad()
                seq_true = seq_true.to(next(self.parameters()).device).to(dtype=float32)
                seq_pred = self.forward(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        predictions, losses = [], []
        with no_grad():
            for seq_true in data:
                seq_true = from_numpy(seq_true).unsqueeze(0).to(next(self.parameters()).device).to(dtype=float32)
                seq_pred = self.forward(seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
            predictions = np.array(predictions)
            predictions = predictions.reshape(predictions.shape[0], self.n_steps, self.n_steps)
        return np.array(predictions)

    def score_samples(self, data):
        train_prediction = self.predict(data)
        residuals = np.abs(data - train_prediction)
        residuals = np.mean(residuals, axis=1).sum(axis=1)
        return residuals
