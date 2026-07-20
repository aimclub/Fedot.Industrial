from dataclasses import dataclass


@dataclass(frozen=True)
class AutoMLBudget:
    time_limit: float  # seconds
    trial_limit: int
    memory_hint: int  # MB
    random_seed: int

    def __post_init__(self):
        if self.time_limit <= 0:
            raise ValueError('time_limit must be positive')
        if self.trial_limit < 0:
            raise ValueError('trial_limit must be non-negative')
        if self.memory_hint <= 0:
            raise ValueError('memory_hint must be positive')


@dataclass(frozen=True)
class AutoMLResult:
    wall_clock: float
    fitted_model_count: int
    failure_count: int
    quantile_support: bool
