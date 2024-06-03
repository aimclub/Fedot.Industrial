import pandas as pd

from typing import Optional
from sklearn.preprocessing import StandardScaler
from torch import Tensor, cuda, device, no_grad
from torch.nn import Conv1d, ConvTranspose1d, Dropout, Module, MSELoss, Sequential, ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly_detector import AnomalyDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class ConvolutionalAutoEncoderDetector(AnomalyDetector):
    """
    A reconstruction convolutional autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.ucl_quantile = self.params.get('ucl_quantile', 0.999)
        self.n_steps_share = self.params.get('n_steps_share', 0.15)
        self.transformation_mode = 'full'
        self.scaler = StandardScaler()

    def build_model(self):
        model_impl = ConvolutionalAutoEncoder(n_steps=self.n_steps).to(device)
        return model_impl

    def fit(self, input_data: InputData) -> None:
        self.n_steps = round(input_data.features.shape[0] * self.n_steps_share)
        converted_input_data = self.convert_input_data(input_data)
        self.model_impl = self.build_model()

        self.model_impl.fit(converted_input_data)
        self.ucl = pd.Series(
            np.abs(converted_input_data - self.model_impl.predict(converted_input_data)).mean(axis=1).sum(axis=1)
        ).quantile(self.ucl_quantile) * 4 / 3

    def predict(self, input_data: InputData):
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        prediction = np.zeros(input_data.target.shape)

        cnn_residuals = pd.Series(
            np.abs(converted_input_data - self.model_impl.predict(converted_input_data)).mean(axis=1).sum(axis=1)
        )
        anomalous_data = cnn_residuals > self.ucl
        anomalous_data_indices = []
        # data i is an anomaly if samples [(i - n_steps + 1) to (i)] are anomalies
        for data_idx in range(self.n_steps - 1, len(converted_input_data) - self.n_steps + 1):
            if np.all(anomalous_data[data_idx - self.n_steps + 1: data_idx]):
                anomalous_data_indices.append(data_idx)

        labels = pd.Series(data=0)
        labels.iloc[anomalous_data_indices] = 1
        labels = labels.values.reshape(-1, 1)
        start_idx, end_idx = prediction.shape[0] - labels.shape[0], prediction.shape[0]
        prediction[np.arange(start_idx, end_idx), :] = labels
        return prediction

    def convert_input_data(self, input_data: InputData, fit_stage: bool = True) -> np.ndarray:
        if fit_stage:
            values = self.scaler.fit_transform(input_data.features)
        else:
            values = self.scaler.transform(input_data.features)
        output = []
        for i in range(len(values) - self.n_steps + 1):
            output.append(values[i: (i + self.n_steps)])
        return np.stack(output)


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
