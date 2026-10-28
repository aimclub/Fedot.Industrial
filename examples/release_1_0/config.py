"""Pure, bounded JSON contract, separate from loading data and fitting models."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


SCENARIOS = ("motions_graph", "airline", "motions_neural")


@dataclass(frozen=True)
class GraphParams:
    max_iter: int
    c: float


@dataclass(frozen=True)
class ForecastParams:
    horizon: int
    window_size: int
    alpha: float


@dataclass(frozen=True)
class NeuralParams:
    steps: int
    width: int
    learning_rate: float
    batch_size: int


@dataclass(frozen=True)
class ReleaseConfig:
    schema: str
    scenario: str
    seed: int
    backend: str
    device: str
    parameters: GraphParams | ForecastParams | NeuralParams

    def to_record(self):
        return asdict(self)


def _number(raw, name, low, high, *, integer=False):
    value = raw[name]
    types = (int,) if integer else (int, float)
    if type(value) not in types or not low <= value <= high or not math.isfinite(value):
        raise ValueError(
            f"{name} must be {'an integer' if integer else 'finite'} in [{low}, {high}]")
    return value


def parse_config(raw):
    fields = {"schema", "scenario", "seed", "backend", "device", "parameters"}
    if not isinstance(raw, dict) or set(raw) != fields or raw["schema"] != "industrial_release_example@1":
        raise ValueError(
            "Config fields/schema must match industrial_release_example@1")
    if raw["scenario"] not in SCENARIOS:
        raise ValueError("Unknown scenario")
    _number(raw, "seed", 0, 2**32 - 1, integer=True)
    if raw["backend"] not in ("sequential", "process") or raw["device"] not in ("cpu", "cuda"):
        raise ValueError("Use sequential/process and cpu/cuda")
    neural = raw["scenario"] == "motions_neural"
    if (neural and raw["backend"] != "sequential") or (not neural and raw["device"] != "cpu"):
        raise ValueError(
            "Neural execution is sequential; graph/forecast execution is CPU only")
    rules = {
        "motions_graph": (GraphParams, {"max_iter": (10, 500, True), "c": (0.001, 100, False)}),
        "airline": (ForecastParams, {"horizon": (1, 24, True), "window_size": (4, 36, True),
                                     "alpha": (0.001, 100, False)}),
        "motions_neural": (NeuralParams, {"steps": (1, 10, True), "width": (4, 16, True),
                                          "learning_rate": (0.00001, 0.1, False),
                                          "batch_size": (4, 40, True)}),
    }
    factory, limits = rules[raw["scenario"]]
    params = raw["parameters"]
    if not isinstance(params, dict) or set(params) != set(limits):
        raise ValueError(f"parameters must contain exactly {tuple(limits)}")
    values = {name: _number(params, name, low, high, integer=integer)
              for name, (low, high, integer) in limits.items()}
    return ReleaseConfig(**{key: value for key, value in raw.items() if key != "parameters"},
                         parameters=factory(**values))


def load_config(scenario="motions_graph", path=None, *, backend=None, device=None):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    source = Path(path) if path else Path(
        __file__).with_name(f"{scenario}.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if backend is not None:
        raw["backend"] = backend
    if device is not None:
        raw["device"] = device
    return parse_config(raw)
