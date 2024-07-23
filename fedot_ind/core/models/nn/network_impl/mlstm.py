from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from typing import Optional
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array


class SqueezeExciteBlock(nn.Module):
    def __init__(self, input_channels, filters, reduce=4):
        super().__init__()
        self.filters = filters
        self.pool = nn.AvgPool1d(input_channels)
        self.bottleneck = max(self.filters // reduce, 4)
        self.fc1 = nn.Linear(self.filters, self.bottleneck, bias=False)
        self.fc2 = nn.Linear(self.bottleneck, self.filters, bias=False)
        torch.nn.init.kaiming_normal_(self.fc1.weight.data)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data)

    def forward(self, x):
        input_x = x
        x = self.pool(x)
        x = F.relu(self.fc1(x.view(-1, 1, self.filters)))
        x = F.sigmoid(self.fc2(x))
        x = x.view(-1, self.filters, 1) * input_x
        return x


class MLSTM_module(nn.Module):
    def __init__(self, input_size, input_channels,
                 inner_size, inner_channels,
                 output_size, num_layers, dropout=0.25):
        super().__init__()
        self.proj = nn.Linear(input_size * inner_channels + input_channels * inner_size, output_size)
        self.lstm = nn.LSTM(input_size, inner_size, num_layers,
                            batch_first=True, dropout=dropout)

        squeeze_excite_size = input_size
        self.conv_branch = nn.Sequential(
            nn.Conv1d(input_channels, inner_channels,
                      padding='same',
                      kernel_size=9),
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(),
            SqueezeExciteBlock(squeeze_excite_size, inner_channels),
            nn.Conv1d(inner_channels, inner_channels * 2,
                      padding='same',
                      kernel_size=5,
                      ),  # c x l | n x c x l
            nn.BatchNorm1d(inner_channels * 2),  # n x c | n x c x l
            nn.ReLU(),
            SqueezeExciteBlock(squeeze_excite_size, inner_channels * 2),
            nn.Conv1d(inner_channels * 2, inner_channels,
                      padding='same',
                      kernel_size=3,
                      ),  # c x l | n x c x l
            nn.BatchNorm1d(inner_channels),  # n x c | n x c x l
            nn.ReLU(),
        )
        seq = next(iter(self.conv_branch.modules()))
        idx = [0, 4, 8]
        for i in idx:
            torch.nn.init.kaiming_uniform_(seq[i].weight.data)

    def forward(self, x, hidden_state=None, return_hidden=False):
        x_lstm, hidden_state = self.lstm(x, hidden_state)  # n x input_ch x inner_size
        x_conv = self.conv_branch(x)  # n x inner_ch x len
        x = torch.cat([torch.flatten(x_lstm, start_dim=1), torch.flatten(x_conv, start_dim=1)], dim=-1)
        x = F.softmax(self.proj(x))
        if return_hidden:
            return x, hidden_state
        return x


class MLSTM(BaseNeuralModel):
    f"""
    The Multivariate Long Short Term Memory Fully Convolutional Network (MLSTM)
    from F. Karim, S. Majumdar, H. Darabi, and S. Harford, “Multivariate LSTM-FCNs for time series classification,” Neural
    Networks, vol. 116, pp. 237–245, 2019.

    {BaseNeuralModel.__doc__}
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.dropout = params.get('dropout', 0.25)
        self.hidden_size = params.get('hidden_size', 64)
        self.hidden_channels = params.get('hidden_channels', 32)
        self.num_layers = params.get('num_layers', 2)
        self.interval_percentage = params.get('interval_percentage', 10)
        self.min_ts_length = params.get('min_ts_length', 5)
        self.fitting_mode = params.get('fitting_mode', 'zero_padding')
        self.proba_thr = params.get('proba_thr', None)

    def __repr__(self):
        return 'MLSTM'

    def _compute_prediction_points(self, n_idx):
        interval_length = max(int(n_idx * self.interval_percentage / 100), self.min_ts_length)
        prediction_idx = np.arange(interval_length - 1, n_idx, interval_length)
        self.earliness = 1 - prediction_idx / n_idx
        return prediction_idx, interval_length

    def _init_model(self, ts: InputData):
        _, input_channels, input_size = ts.features.shape
        self.input_size = input_size
        self.prediction_idx, self.interval = self._compute_prediction_points(input_size)
        self.model = MLSTM_module(input_size if self.fitting_mode != 'moving_window' else self.interval,
                                  input_channels,
                                  self.hidden_size, self.hidden_channels,
                                  self.num_classes, self.num_layers,
                                  self.dropout)
        self.model_for_inference = MLSTM_module(input_size if self.fitting_mode != 'moving_window' else self.interval,
                                                input_channels,
                                                self.hidden_size, self.hidden_channels,
                                                self.num_classes, self.num_layers,
                                                self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = CROSS_ENTROPY()
        return loss_fn, optimizer

    @convert_to_3d_torch_array
    def _fit_model(self, ts: InputData):
        mode = self.fitting_mode
        loss_fn, optimizer = self._init_model(ts)
        train_loader, val_loader = self._prepare_data(ts, split_data=True,
                                                      collate_fn=getattr(self, '_augment_with_zeros'))
        if mode == 'zero_padding':
            super()._train_loop(
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer
            )
        elif mode == 'moving_window':
            self._train_loop(
                train_loader=train_loader,
                val_loader=None,
                loss_fn=loss_fn,
                optimizer=optimizer
            )
        else:
            raise ValueError('Unknown fitting mode')

    def _moving_window_output(self, inputs):
        hidden_state = None
        output = -torch.ones((inputs.shape[0], self.num_classes))
        for i in self.prediction_idx:
            if i >= inputs.shape[-1]:
                break
            batch_interval = inputs[..., i - self.prediction_idx[0]: i + 1]
            output, hidden_state = self.model(batch_interval, hidden_state, return_hidden=True)
        return output

    def _train_one_batch(self, batch, optimizer, loss_fn):
        if self.fitting_mode == 'zero_padding':
            return super()._train_one_batch(batch, optimizer, loss_fn)
        elif self.fitting_mode == 'moving_window':
            optimizer.zero_grad()
            inputs, targets = batch
            output = self._moving_window_output(inputs)
            loss = loss_fn(output, targets.float())
            loss.backward()
            optimizer.step()
            training_loss = loss.data.item() * inputs.size(0)
            total = targets.size(0)
            correct = (torch.argmax(output, 1) ==
                       torch.argmax(targets, 1)).sum().item()
            return training_loss, total, correct
        else:
            raise ValueError('Unknown fitting mode!')

    def _eval_one_batch(self, batch, loss_fn):
        if self.fitting_mode == 'zero_padding':
            return super()._eval_one_batch(batch, loss_fn)
        elif self.fitting_mode == 'moving_window':
            inputs, targets = batch
            output = self._moving_window_output(inputs)
            loss = loss_fn(output, targets.float())
            valid_loss = loss.data.item() * inputs.size(0)
            total = targets.size(0)
            correct = (torch.argmax(output, 1) ==
                       torch.argmax(targets, 1)).sum().item()
            return valid_loss, total, correct
        else:
            raise ValueError('Unknown fitting mode!')

    @convert_to_3d_torch_array
    def _predict_model(self, x_test: InputData, output_mode: str = 'default'):
        self.model.eval()
        if self.fitting_mode == 'zero_padding':
            x_test = self._padding(x_test).to(self._device)
            pred = self.model(x_test)
        elif self.fitting_mode == 'moving_window':
            pred = self._moving_window_output(torch.tensor(x_test).float())
        else:
            raise ValueError('Unknown prediction mode')
        pred = pred.detach()
        return self._convert_predict(pred, output_mode)

    def _padding(self, ts: np.array):
        if ts.shape[-1] == self.input_size:
            return torch.tensor(ts).float()
        n, ch, size = ts.shape
        x = torch.zeros((n, ch, self.input_size)).float()
        x[..., :size] = ts
        return x

    def _augment_with_zeros(self, batch: np.array):
        X, y = zip(*batch)
        X, y = np.stack(X), np.stack(y)
        X_res, y_res = [], []
        for i in self.prediction_idx:
            x = X[...]
            x[..., :i + i] = 0
            X_res.append(x)
            y_res.append(y)
        X_res = np.concatenate(X_res)
        y_res = np.concatenate(y_res)
        perm = np.random.permutation(X_res.shape[0])
        return torch.tensor(X_res[perm]), torch.tensor(y_res[perm])

    def _transform_score(self, probas):
        # linear interp
        thr = self.proba_thr
        probas = probas - thr
        positive = probas > 0
        probas[positive] *= 1 / (1 - thr)
        probas[~positive] *= 1 / thr
        return probas
