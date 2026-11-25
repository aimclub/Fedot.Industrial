"""Jointly trained low-rank prediction from observed prefixes of Page blocks."""

import numpy as np
import torch
from torch import nn

from fedot_ind.neural.training import TrainingPlan, TrainingSession

from ..page_models import _observed_projection, page_forecast_examples


class PageProjection(nn.Module):
    """Input is (batch, L-1), with targets external to forward."""

    def __init__(self, basis, *, learn_projection, baseline_lag=1, center=0., scale=1.):
        super().__init__()
        basis = torch.as_tensor(np.array(basis, copy=True), dtype=torch.float32)
        if basis.ndim != 2 or not 1 <= baseline_lag <= len(basis):
            raise ValueError("The baseline must use an available lag.")
        self.projection = nn.Parameter(basis, requires_grad=learn_projection)
        self.head = nn.Linear(basis.shape[1], 1)
        # Starting both controls at the same baseline isolates the projection.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.baseline_lag = baseline_lag
        self.register_buffer("center", torch.as_tensor(center, dtype=basis.dtype))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=basis.dtype))

    def forward(self, history):
        if history.ndim != 2 or history.shape[1] != self.projection.shape[0]:
            raise ValueError("Expected (batch, L-1) observed prefixes.")
        features = ((history - self.center) / self.scale) @ self.projection
        return history[:, -self.baseline_lag] + self.scale * self.head(features).squeeze(-1)


def fit_page_projection(series, *, length, rank, learn_projection, baseline_lag=1,
                        epochs=16, learning_rate=0.01):
    features, target = page_forecast_examples(series, length)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("The neural prototype requires fully observed training blocks.")
    center, scale = float(features.mean()), float(features.std() or 1.)
    _, basis, _ = _observed_projection((features - center) / scale, rank)
    with torch.random.fork_rng(devices=[]):
        model = PageProjection(basis, learn_projection=learn_projection, baseline_lag=baseline_lag,
                               center=center, scale=scale)
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32))
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32)
    session = TrainingSession(TrainingPlan(epochs=epochs), model, torch.optim.Adam(
        model.parameters(), lr=learning_rate), nn.MSELoss(), batches)
    session.run()
    model.load_state_dict(session.best_weights())
    return model.eval()
