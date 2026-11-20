"""Segment and amortised WIRE experiments with a shared training loop."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn

from fedot_ind.neural.training import TrainingPlan, TrainingSession

from ..grid import positive_count
from ..wavelets import _finite_series


def _train(model, features, target, epochs, lr):
    dataset = torch.utils.data.TensorDataset(features, target)
    session = TrainingSession(TrainingPlan(epochs=epochs), model, torch.optim.Adam(model.parameters(), lr=lr),
                              nn.MSELoss(), torch.utils.data.DataLoader(dataset, batch_size=32))
    session.run()
    model.load_state_dict(session.best_weights())
    return model.eval()


class _Gabor(nn.Module):
    def forward(self, value):
        return torch.cos(8 * value) * torch.exp(-value.square())


@dataclass(frozen=True, eq=False)
class SegmentForecast:
    values: np.ndarray
    fit_seconds: float
    history_rmse: float


def predict_segment_wire(history, horizon, *, width=16, epochs=12, lr=.01, seed=0):
    x = _finite_series(history)
    horizon, width = positive_count(horizon, "horizon"), positive_count(width, "width")
    if len(x) < 2:
        raise ValueError("A coordinate scaling needs at least two known observations.")
    t = torch.arange(len(x) + horizon, dtype=torch.float32).reshape(-1, 1) / (len(x) - 1)
    center, scale = x.mean(), x.std() or 1.
    target = torch.as_tensor((x - center) / scale, dtype=torch.float32).reshape(-1, 1)
    start = perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _train(nn.Sequential(nn.Linear(1, width), _Gabor(), nn.Linear(width, 1)),
                       t[:len(x)], target, epochs, lr)
    elapsed = perf_counter() - start
    with torch.no_grad():
        values = (model(t).reshape(-1).numpy() * scale + center).astype(float)
    return SegmentForecast(values[len(x):], elapsed, float(np.sqrt(np.mean((values[:len(x)] - x)**2))))


class AmortizedWIRE(nn.Module):
    """Encode context into weights of a shared coordinate basis; forward never fits."""

    def __init__(self, context, horizon, *, width=16, atoms=8, localize=True):
        super().__init__()
        self.context = positive_count(context, "context")
        horizon, atoms = positive_count(horizon, "horizon"), positive_count(atoms, "atoms")
        if self.context < 2:
            raise ValueError("At least two observations are required.")
        time = torch.arange(context, context + horizon, dtype=torch.float32) / (context - 1)
        phase = (time[:, None] - 1) * torch.arange(1, atoms + 1, dtype=time.dtype)[None, :]
        envelope = torch.exp(-(time[:, None] - 1).square()) if localize else 1.
        basis = torch.cat((torch.sin(phase) * envelope, torch.cos(phase) * envelope, time[:, None] * 0 + 1), dim=-1)
        self.register_buffer("basis", basis)
        self.encoder = nn.Sequential(nn.Linear(context, width), nn.Tanh(), nn.Linear(width, basis.shape[1]))

    def forward(self, context):
        if context.ndim != 2 or context.shape[1] != self.context:
            raise ValueError("Expected a batch of fixed-length histories.")
        last = context[:, -1:]
        return last + self.encoder(context - last) @ self.basis.T


def fit_amortized_wire(series, *, context, horizon, width=16, atoms=8, epochs=12, lr=.01, seed=0, localize=True):
    context, horizon = positive_count(context, "context"), positive_count(horizon, "horizon")
    windows = [np.lib.stride_tricks.sliding_window_view(_finite_series(item), context + horizon)
               for item in series]
    if not windows:
        raise ValueError("At least one training series is required.")
    batch = torch.as_tensor(np.concatenate(windows), dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _train(AmortizedWIRE(context, horizon, width=width, atoms=atoms, localize=localize),
                      batch[:, :context], batch[:, context:], epochs, lr)
