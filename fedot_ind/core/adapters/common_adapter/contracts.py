from dataclasses import dataclass
import pandas as pd

@dataclass
class AutoMLBudget:
    time_limit: float  # seconds
    trial_limit: int
    memory_hint: int  # MB
    random_seed: int


@dataclass
class AutoMLResult:
    wall_clock: float
    fitted_model_count: int
    failure_count: int
    quantile_support: bool
