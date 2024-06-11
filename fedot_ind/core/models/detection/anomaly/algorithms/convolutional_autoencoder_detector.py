from collections import OrderedDict
from typing import Optional

from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor, cuda, device, no_grad
from torch.nn import Conv1d, ConvTranspose1d, Module, MSELoss, Sequential, ReLU
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.models.detection.anomaly.algorithms.autoencoder_detector import AutoEncoderDetector

device = device("cuda:0" if cuda.is_available() else "cpu")


class ConvolutionalAutoEncoderDetector(AutoEncoderDetector):
    """A reconstruction convolutional autoencoder model to detect anomalies
    in timeseries data using reconstruction error as an anomaly score.
    """

    def build_model(self):
        CAE_params = dict(n_steps=self.n_steps,
                          learning_rate=self.learning_rate)
        return ConvolutionalAutoEncoder(CAE_params).to(device)


class ConvolutionalAutoEncoder(Module):
    def __init__(self,
                 params: Optional[OperationParameters] = None):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.learning_rate = params.get('learning_rate', 0.001)
        self.n_steps = params.get('n_steps', 10)
        self.encoder_layers = params.get('num_encoder_layers', 2)
        self.decoder_layers = params.get('num_decoder_layers', 2)
        self.latent_layer_params = params.get('latent_layer', 16)
        self.convolutional_params = params.get('convolutional_params',
                                               dict(kernel_size=7, stride=2, padding=3))
        self.activation_func = params.get('act_func', ReLU)
        self.dropout_rate = params.get('dropout_rate', 0.5)
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        encoder_layer_dict = OrderedDict()
        for i in range(self.encoder_layers):
            if i == 0:
                in_channels = self.n_steps
                out_channels = self.latent_layer_params
            elif i == self.encoder_layers - 1:
                out_channels = self.latent_layer_params
            else:
                out_channels = 32
            encoder_layer_dict.update({f'conv{i}': Conv1d(in_channels=in_channels,
                                                          out_channels=out_channels,
                                                          **self.convolutional_params)})
            encoder_layer_dict.update({f'relu{i}': self.activation_func()})
            in_channels = out_channels
        self.encoder = Sequential(encoder_layer_dict)

    def _build_decoder(self):
        decoder_layer_dict = OrderedDict()
        for i in range(self.decoder_layers):
            if i == 0:
                in_channels = self.latent_layer_params
                out_channels = self.n_steps
            elif i == self.encoder_layers - 1:
                out_channels = self.n_steps
            else:
                out_channels = 32

            decoder_layer_dict.update({f'conv{i}':
                                           ConvTranspose1d(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           output_padding=1, **self.convolutional_params)})
            decoder_layer_dict.update({f'relu{i}': self.activation_func()})
            in_channels = out_channels
        self.decoder = Sequential(decoder_layer_dict)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self,
            data,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1):
        dataset = TensorDataset(Tensor(data))
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
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
            print(f'Loss- {loss}')
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

    def score_samples(self, data):
        train_prediction = self.predict(data)
        residuals = np.abs(data - train_prediction)
        residuals = np.mean(residuals, axis=1).sum(axis=1)
        return residuals
