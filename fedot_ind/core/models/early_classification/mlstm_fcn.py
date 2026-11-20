"""Jointly trained LSTM/FCN at a fixed multichannel observation budget."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fedot_ind.neural.training import TrainingPlan, TrainingSession

from .protocol import EarlyDecision, training_inputs


class MLSTMFCN(nn.Module):
    def __init__(self, channels, cutoff, classes, *, width=16):
        super().__init__()
        self.input_shape = (channels, cutoff)
        self.recurrent = nn.LSTM(cutoff, width, batch_first=True)
        self.convolution = nn.Sequential(
            nn.Conv1d(channels, width, 7, padding=3), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        reduced = max(1, width // 4)
        self.channel_gate = nn.Sequential(nn.Linear(width, reduced), nn.ReLU(), nn.Linear(reduced, width), nn.Sigmoid())
        self.head = nn.Linear(2 * width, classes)

    def forward(self, prefix):
        if prefix.ndim != 3 or tuple(prefix.shape[1:]) != self.input_shape:
            raise ValueError("Expected (batch, channels, cutoff) inputs.")
        # MLSTM-FCN uses dimension shuffle: recurrent steps run over channels.
        _, (last, _) = self.recurrent(prefix)
        features = self.convolution(prefix).mean(dim=-1)
        return self.head(torch.cat((last[-1], features * self.channel_gate(features)), dim=1))


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


def fit_mlstm_fcn(series, target, schedule, *, cutoff, width=16, epochs=12, lr=.005, seed=0):
    x, y = training_inputs(series, target, schedule)
    if cutoff not in schedule.cutoffs:
        raise ValueError("Select a declared fixed cutoff on training data.")
    features = torch.from_numpy(x[:, :, :cutoff].astype(np.float32))
    labels = torch.tensor([schedule.classes.index(item) for item in y])
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MLSTMFCN(schedule.channels, cutoff, len(schedule.classes), width=width)
        session = TrainingSession(TrainingPlan(epochs=epochs), model, torch.optim.Adam(model.parameters(), lr=lr),
                                  nn.CrossEntropyLoss(), batches)
        session.run()
        model.load_state_dict(session.best_weights())
    return FixedPrefixMLSTM(model.eval(), schedule.classes)
