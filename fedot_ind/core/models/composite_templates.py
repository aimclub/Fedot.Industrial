"""Configuration and topology of atomic Industrial search operations."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real

from industrial_forecasting.dwt import ArinoSpec
from industrial_forecasting.grid import positive_count
from industrial_forecasting.wavelet_hfcm import WaveletHFCMSpec


class CompositeHorizonError(ValueError):
    """A configured or requested horizon is incompatible with a composite."""


class ForecastHorizonPolicy(Enum):
    ANY_POSITIVE = "any_positive"
    ONE_STEP = "one_step"

    def resolve(self, parameters, task_horizon):
        try:
            configured = (positive_count(parameters["forecast_length"], "forecast_length")
                          if "forecast_length" in parameters else None)
            actual = positive_count(task_horizon, "Task forecast horizon") if task_horizon is not None else None
        except ValueError as error:
            raise CompositeHorizonError(str(error)) from error
        if configured is not None and actual is not None and configured != actual:
            raise CompositeHorizonError("forecast_length must agree with the task forecast horizon.")
        horizon = actual if actual is not None else configured
        if self is ForecastHorizonPolicy.ONE_STEP and horizon not in (None, 1):
            raise CompositeHorizonError("Page projection requires a one-step forecast horizon.")
        return horizon


@dataclass(frozen=True)
class CompositeTemplate:
    operation: str
    version: int
    edges: tuple[tuple[str, str], ...]
    config_type: type
    fields: tuple[tuple[str, str], ...]
    forecast_horizon: ForecastHorizonPolicy | None = None

    @property
    def parameters(self):
        return tuple(key for key, _ in self.fields)

    @property
    def roles(self):
        return tuple(sorted({key.split(".")[0] for key in self.parameters}))

    def flatten(self, params):
        result = {}
        roles = self.roles
        for key, value in dict(params or {}).items():
            if key in roles:
                if not isinstance(value, Mapping):
                    raise ValueError(f"Composite role {key!r} requires a parameter mapping.")
                entries = ((f"{key}.{leaf}", item) for leaf, item in value.items())
            else:
                entries = ((key, value),)
            for name, item in entries:
                if name in result:
                    raise ValueError(f"Repeated composite parameter: {name}")
                result[name] = item
        return result

    def bind(self, params, *, task_horizon=None):
        config = self.flatten(params)
        optional = {"forecast_length"} if self.forecast_horizon is not None else set()
        unknown = config.keys() - set(self.parameters) - optional
        if unknown:
            raise ValueError(f"Unknown composite parameters: {sorted(unknown)}")
        if self.forecast_horizon is not None:
            self.forecast_horizon.resolve(config, task_horizon)
        return self.config_type(**{field: config[key] for key, field in self.fields if key in config})

    def describe(self, params):
        config = self.bind(params)
        return {"template": self.operation, "version": self.version, "edges": self.edges,
                "parameters": {key: getattr(config, field) for key, field in self.fields}}


@dataclass(frozen=True)
class NeuralCompositeConfig:
    epochs: int = 4
    learning_rate: float = .005

    def __post_init__(self):
        positive_count(self.epochs, "epochs")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real)
                or not isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("Learning rate must be finite and positive.")


@dataclass(frozen=True)
class CWTCompositeConfig(NeuralCompositeConfig):
    window: int = 24
    width: int = 12
    cnn_width: int | None = None
    lstm_width: int | None = None
    attention: bool = True

    def __post_init__(self):
        super().__post_init__()
        if positive_count(self.window, "window") < 2 or type(self.attention) is not bool:
            raise ValueError("CWT needs at least two observations and a boolean attention flag.")
        positive_count(self.width, "width")
        for key in ("cnn_width", "lstm_width"):
            if getattr(self, key) is not None:
                positive_count(getattr(self, key), key)


@dataclass(frozen=True)
class PageCompositeConfig(NeuralCompositeConfig):
    length: int = 12
    rank: int = 2
    baseline_lag: int = 1
    learn_projection: bool = True

    def __post_init__(self):
        super().__post_init__()
        width = positive_count(self.length, "length") - 1
        if (positive_count(self.rank, "rank") > width
                or positive_count(self.baseline_lag, "baseline_lag") > width
                or type(self.learn_projection) is not bool):
            raise ValueError("Page rank and residual lag must fit the observed prefix.")


@dataclass(frozen=True)
class MLSTMCompositeConfig(NeuralCompositeConfig):
    cutoff: int = 16
    width: int = 12
    fcn_width: int | None = None
    lstm_width: int | None = None

    def __post_init__(self):
        super().__post_init__()
        positive_count(self.cutoff, "cutoff")
        positive_count(self.width, "width")
        for key in ("fcn_width", "lstm_width"):
            if getattr(self, key) is not None:
                positive_count(getattr(self, key), key)


HFCM = CompositeTemplate(
    "wavelet_hfcm_forecaster", 1,
    (("history", "haar"), ("haar", "lags"), ("lags", "hfcm"), ("hfcm", "sum")),
    WaveletHFCMSpec, (("haar.levels", "levels"), ("hfcm.order", "order"), ("hfcm.alpha", "alpha")),
    forecast_horizon=ForecastHorizonPolicy.ANY_POSITIVE,
)
ARINO = CompositeTemplate(
    "arino_forecaster", 1,
    (("history", "pad"), ("pad", "dwt"), ("dwt", "low"), ("dwt", "high"),
     ("low", "trend"), ("high", "seasonal"), ("trend", "sum"), ("seasonal", "sum")),
    ArinoSpec, (("pad.length", "padding"), ("dwt.wavelet", "wavelet"), ("dwt.levels", "levels"),
                ("dwt.slow_bands", "slow_bands"), ("trend.lookback", "trend_lookback"),
                ("seasonal.period", "seasonal_period")),
    forecast_horizon=ForecastHorizonPolicy.ANY_POSITIVE,
)
_TRAIN = (("training.epochs", "epochs"), ("training.learning_rate", "learning_rate"))
CWT = CompositeTemplate(
    "cwt_attention_forecaster", 1,
    (("history", "cwt"), ("cwt", "cnn"), ("history", "lstm"), ("lstm", "attention"),
     ("cnn", "attention"), ("attention", "head"), ("lstm", "head")),
    CWTCompositeConfig, (("cwt.window", "window"), ("encoders.width", "width"),
                         ("cnn.width", "cnn_width"), ("lstm.width", "lstm_width"),
                         ("attention.enabled", "attention"), *_TRAIN),
    forecast_horizon=ForecastHorizonPolicy.ANY_POSITIVE,
)
PAGE = CompositeTemplate(
    "page_projection_forecaster", 1,
    (("history", "page"), ("page", "projection"), ("projection", "head"),
     ("history", "residual"), ("residual", "sum"), ("head", "sum")),
    PageCompositeConfig, (("page.length", "length"), ("projection.rank", "rank"),
                          ("projection.learn", "learn_projection"), ("residual.lag", "baseline_lag"), *_TRAIN),
    forecast_horizon=ForecastHorizonPolicy.ONE_STEP,
)
MLSTM = CompositeTemplate(
    "mlstm_fcn_classifier", 1,
    (("prefix", "lstm"), ("prefix", "fcn"), ("lstm", "head"), ("fcn", "head")),
    MLSTMCompositeConfig, (("prefix.cutoff", "cutoff"), ("encoders.width", "width"),
                           ("fcn.width", "fcn_width"), ("lstm.width", "lstm_width"), *_TRAIN),
)
TEMPLATES = {item.operation: item for item in (ARINO, HFCM, CWT, PAGE, MLSTM)}


def template_spec(operation, params, *, task_horizon=None):
    return TEMPLATES[operation].bind(params, task_horizon=task_horizon)
