"""Joint time/CWT encoders with an explicit point or direction output."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fedot_ind.neural.training import SupervisedBatch, TrainingPlan, TrainingSession

from ..dwt import _wavelet
from ..grid import positive_count
from ..wavelets import _finite_series


@dataclass(frozen=True)
class WaveletAttentionSpec:
    window: int = 24
    horizon: int = 1
    scales: tuple[float, ...] = (1., 2., 4., 8.)
    wavelet: str = "morl"
    width: int = 12
    task: str = "point"
    attention: bool = True
    branch: str = "joint"
    cnn_width: int | None = None
    lstm_width: int | None = None

    def __post_init__(self):
        positive_count(self.window, "window")
        positive_count(self.horizon, "horizon")
        positive_count(self.width, "width")
        for key in ("cnn_width", "lstm_width"):
            if getattr(self, key) is not None:
                positive_count(getattr(self, key), key)
        scales = np.asarray(self.scales, dtype=float)
        if (self.window < 2 or scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all()
                or np.any(scales <= 0) or np.any(np.diff(scales) <= 0)
                or self.task not in {"point", "direction"} or self.branch not in {"joint", "cnn", "lstm"}):
            raise ValueError("Invalid observation window, increasing CWT scales or output task.")


def cwt_view(windows, spec):
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != spec.window or not len(x) or not np.isfinite(x).all():
        raise ValueError("CWT needs finite (batch, window) input.")
    raw = torch.from_numpy(x[:, :, None].copy())
    if spec.branch == "lstm":
        return raw, torch.empty((len(x), 0))
    coefficients, _ = _wavelet().cwt(x, spec.scales, spec.wavelet, axis=-1)
    maps = np.stack((coefficients.real, coefficients.imag), axis=1).transpose(2, 1, 0, 3)
    return raw, torch.from_numpy(maps.astype(np.float32, copy=False))


class WaveletAttention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cnn_width, lstm_width = spec.cnn_width or spec.width, spec.lstm_width or spec.width
        self.temporal = None if spec.branch == "cnn" else nn.LSTM(1, lstm_width, batch_first=True)
        self.spectral = (
            None if spec.branch == "lstm" else nn.Sequential(nn.Conv2d(2, cnn_width, 3, padding=1), nn.ReLU())
        )
        if spec.branch == "joint" and spec.attention:
            self.location_score = nn.Linear(cnn_width, spec.width)
            self.history_score = nn.Linear(lstm_width, spec.width, bias=False)
            self.attention_score = nn.Linear(spec.width, 1, bias=False)
        hidden = (cnn_width if self.spectral is not None else 0) + (lstm_width if self.temporal is not None else 0)
        self.head = nn.Linear(hidden, 2 if spec.task == "direction" else spec.horizon)

    def forward(self, views):
        raw, scalogram = views
        if self.temporal is not None:
            _, (hidden, _) = self.temporal(raw)
            history = hidden[-1]
            if self.spectral is None:
                return self.head(history)
        patches = self.spectral(scalogram).flatten(2).transpose(1, 2)
        if self.temporal is None:
            return self.head(patches.mean(dim=1))
        if self.spec.attention:
            logits = self.attention_score(torch.relu(
                self.location_score(patches) + self.history_score(history[:, None]),
            ))
            summary = (patches * logits.softmax(dim=1)).sum(dim=1)
        else:
            summary = patches.mean(dim=1)
        return self.head(torch.cat((summary, history), dim=1))


@dataclass(frozen=True, eq=False)
class FittedWaveletAttention:
    model: WaveletAttention
    center: float
    scale: float

    @torch.inference_mode()
    def predict(self, history):
        x = _finite_series(history)[-self.model.spec.window:]
        result = self.model(cwt_view(((x - self.center) / self.scale)[None], self.model.spec))[0]
        if self.model.spec.task == "direction":
            return result.softmax(dim=0).numpy()
        return result.numpy() * self.scale + self.center


def fit_wavelet_attention(histories, spec=WaveletAttentionSpec(), *, epochs=10, lr=.005, seed=0):
    series = tuple(_finite_series(x) for x in histories)
    if not series or any(len(x) < spec.window + spec.horizon for x in series):
        raise ValueError("Each training history must contain a complete input and target horizon.")
    values = np.concatenate(series)
    center, scale = values.mean(), values.std() or 1.
    examples = np.concatenate([np.lib.stride_tricks.sliding_window_view(
        (x - center) / scale, spec.window + spec.horizon,
    ) for x in series])
    history, future = examples[:, :spec.window], examples[:, spec.window:]
    raw, maps = cwt_view(history, spec)
    if spec.task == "direction":
        target = torch.from_numpy((future[:, -1] >= history[:, -1]).astype(np.int64))
        loss = nn.CrossEntropyLoss()
    else:
        target = torch.from_numpy(future.astype(np.float32))
        loss = nn.MSELoss()
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(raw, maps, target), batch_size=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = WaveletAttention(spec)
        session = TrainingSession(TrainingPlan(epochs=epochs), model, torch.optim.Adam(model.parameters(), lr=lr),
                                  loss, batches, batch_adapter=lambda b: SupervisedBatch((b[0], b[1]), b[2]))
        session.run()
        model.load_state_dict(session.best_weights())
    return FittedWaveletAttention(model.eval(), float(center), float(scale))
