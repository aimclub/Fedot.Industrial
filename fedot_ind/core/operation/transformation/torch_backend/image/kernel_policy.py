"""Measured kernel dispatch defaults; no runtime Torch settings are mutated."""
import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def cuda_broadcast_max_boundaries() -> int:
    raw = json.loads(files(__package__).joinpath("kernel_defaults.json").read_text(encoding="utf-8"))
    value = raw.get("cuda_broadcast_max_boundaries")
    if set(raw) != {"cuda_broadcast_max_boundaries"} or type(value) is not int or value < 0:
        raise ValueError("Invalid image kernel defaults")
    return value
