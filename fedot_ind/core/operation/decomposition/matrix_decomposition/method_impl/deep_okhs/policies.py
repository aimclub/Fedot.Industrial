from dataclasses import dataclass


@dataclass(frozen=True)
class RegularizationPolicy:
    base_jitter: float = 1e-8
    condition_threshold: float = 1e10
    fallback_solver: str = "pinv"


@dataclass(frozen=True)
class StabilityPolicy:
    threshold: float = 0.0
    drop_positive_real_modes: bool = True
    sorting_strategy: str = "abs_desc"
