"""Jointly trained LSTM/FCN at a fixed multichannel observation budget."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fedot_ind.neural.training import TrainingPlan, TrainingSession

from .protocol import EarlyDecision, training_inputs


class MLSTMFCN(nn.Module):
    def __init__(self, channels, cutoff, classes, *, width=16, branch="joint", fcn_width=None, lstm_width=None):
        super().__init__()
        if branch not in {"joint", "lstm", "fcn"}:
            raise ValueError("Expected joint, lstm or fcn branch.")
        self.input_shape = (channels, cutoff)
        fcn_width, lstm_width = fcn_width or width, lstm_width or width
        self.recurrent = None if branch == "fcn" else nn.LSTM(cutoff, lstm_width, batch_first=True)
        self.convolution = None if branch == "lstm" else nn.Sequential(
            nn.Conv1d(channels, fcn_width, 7, padding=3), nn.ReLU(),
            nn.Conv1d(fcn_width, fcn_width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(fcn_width, fcn_width, 3, padding=1), nn.ReLU(),
        )
        reduced = max(1, fcn_width // 4)
        self.channel_gate = nn.Sequential(
            nn.Linear(fcn_width, reduced), nn.ReLU(), nn.Linear(reduced, fcn_width), nn.Sigmoid(),
        )
        hidden = (fcn_width if self.convolution is not None else 0) + (lstm_width if self.recurrent is not None else 0)
        self.head = nn.Linear(hidden, classes)

    def forward(self, prefix):
        if prefix.ndim != 3 or tuple(prefix.shape[1:]) != self.input_shape:
            raise ValueError("Expected (batch, channels, cutoff) inputs.")
        # MLSTM-FCN uses dimension shuffle: recurrent steps run over channels.
        if self.recurrent is not None:
            _, (last, _) = self.recurrent(prefix)
            if self.convolution is None:
                return self.head(last[-1])
        features = self.convolution(prefix).mean(dim=-1)
        features = features * self.channel_gate(features)
        if self.recurrent is None:
            return self.head(features)
        return self.head(torch.cat((last[-1], features), dim=1))


@dataclass(frozen=True, eq=False)
class FixedPrefixMLSTM:
    network: MLSTMFCN
    classes: tuple

    @torch.inference_mode()
    def predict_proba(self, series):
        cutoff = self.network.input_shape[1]
        prefix = torch.as_tensor(np.asarray(series)[:, :, :cutoff], dtype=torch.float32)
        if not torch.isfinite(prefix).all():
            raise ValueError("Observed prefix contains nonfinite values.")
        return self.network(prefix).softmax(dim=-1).numpy()

    def predict_one(self, series):
        choice = int(self.predict_proba(np.asarray(series)[None])[0].argmax())
        return EarlyDecision(self.classes[choice], self.network.input_shape[1], False)


def fit_mlstm_fcn(series, target, schedule, *, cutoff, width=16, epochs=12, lr=.005, seed=0, branch="joint",
                  fcn_width=None, lstm_width=None):
    x, y = training_inputs(series, target, schedule)
    if cutoff not in schedule.cutoffs:
        raise ValueError("Select a declared fixed cutoff on training data.")
    features = torch.from_numpy(x[:, :, :cutoff].astype(np.float32))
    labels = torch.tensor([schedule.classes.index(item) for item in y])
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MLSTMFCN(schedule.channels, cutoff, len(schedule.classes), width=width, branch=branch,
                         fcn_width=fcn_width, lstm_width=lstm_width)
        session = TrainingSession(TrainingPlan(epochs=epochs), model, torch.optim.Adam(model.parameters(), lr=lr),
                                  nn.CrossEntropyLoss(), batches)
        session.run()
        model.load_state_dict(session.best_weights())
    return FixedPrefixMLSTM(model.eval(), schedule.classes)
