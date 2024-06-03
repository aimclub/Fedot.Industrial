from torch import Tensor, cuda, device, no_grad
from torch.nn import Conv1d, ConvTranspose1d, Dropout, Module, MSELoss, Sequential, ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly.algorithms.autoencoder_detector import AutoEncoderDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class ConvolutionalAutoEncoderDetector(AutoEncoderDetector):
    """
    A reconstruction convolutional autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.
    """

    def build_model(self):
        return ConvolutionalAutoEncoder(n_steps=self.n_steps).to(device)


class ConvolutionalAutoEncoder(Module):
    def __init__(self, n_steps: int):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.encoder = Sequential(
            Conv1d(in_channels=n_steps, out_channels=32, kernel_size=7, stride=2, padding=3),
            ReLU(),
            Dropout(0.2),
            Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            ReLU()
        )

        self.decoder = Sequential(
            ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3, output_padding=1),
            ReLU(),
            Dropout(0.2),
            ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3, output_padding=1),
            ReLU(),
            ConvTranspose1d(in_channels=32, out_channels=n_steps, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, data, epochs=100, batch_size=32, validation_split=0.1):
        dataset = TensorDataset(Tensor(data))
        optimizer = Adam(self.parameters(), lr=0.001)
        criterion = MSELoss()

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(validation_split * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch[0])
                loss = criterion(outputs, batch[0])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            valid_loss = 0.0
            with no_grad():
                for batch in valid_loader:
                    outputs = self.forward(batch[0])
                    loss = criterion(outputs, batch[0])
                    valid_loss += loss.item()

    def predict(self, data):
        self.eval()
        with no_grad():
            data_torch = Tensor(data)
            predictions = self.forward(data_torch)
            return predictions.numpy()
