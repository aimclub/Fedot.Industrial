from torch import cuda, device, no_grad, float32, from_numpy
from torch.nn import LSTM, Module, MSELoss, Linear

from torch.optim import Adam
from torch.utils.data import DataLoader

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly.algorithms.autoencoder_detector import AutoEncoderDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class LSTMAutoEncoderDetector(AutoEncoderDetector):
    """
    A reconstruction sequence-to-sequence (LSTM-based) autoencoder model to detect anomalies in timeseries data
    using reconstruction error as an anomaly score.
    """

    def build_model(self):
        return LSTMAutoEncoder(n_steps=self.n_steps, n_features=8).to(device)


class LSTMEncoder(Module):
    def __init__(self, n_steps, n_features, embedding_dim=100):
        super(LSTMEncoder, self).__init__()
        self.n_steps, self.n_features = n_steps, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.lstm1 = LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (hidden_n, _) = self.lstm1(x)
        return hidden_n


class LSTMDecoder(Module):
    def __init__(self, n_steps, input_dim=100, n_features=1):
        super(LSTMDecoder, self).__init__()
        self.n_steps, self.input_dim = n_steps, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.lstm1 = LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = Linear(input_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.n_steps, 1, 1).transpose(0, 1)
        x, _ = self.lstm1(x)
        return self.output_layer(x)


class LSTMAutoEncoder(Module):
    def __init__(self, n_steps, n_features, embedding_dim=100):
        super(LSTMAutoEncoder, self).__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.encoder = LSTMEncoder(self.n_steps, self.n_features, embedding_dim)
        self.decoder = LSTMDecoder(self.n_steps, embedding_dim, self.n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, dataset, epochs=100, batch_size=32, val_ratio=0.1):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        criterion = MSELoss()
        optimizer = Adam(self.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for seq_true in train_loader:
                optimizer.zero_grad()
                seq_true = seq_true.to(next(self.parameters()).device).to(dtype=float32)
                seq_pred = self.forward(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()

    def predict(self, dataset):
        predictions, losses = [], []
        with no_grad():
            for seq_true in dataset:
                seq_true = from_numpy(seq_true).unsqueeze(0).to(next(self.parameters()).device).to(dtype=float32)
                seq_pred = self.forward(seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
            predictions = np.array(predictions)
            predictions = predictions.reshape(predictions.shape[0], self.n_steps, self.n_features)
        return np.array(predictions)
