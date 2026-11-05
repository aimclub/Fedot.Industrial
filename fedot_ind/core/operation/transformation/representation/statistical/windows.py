"""One explicit last-axis window contract shared by NumPy and Torch statistics."""
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class StatisticalWindowPlan:
    width: int
    starts: tuple[int, ...]


def statistical_window_plan(length, share, stride, sliding):
    for value, name, low, high in [(length, 'length', 1, None), (share, 'window_size', 1, 100),
                                   (stride, 'stride', 1, None)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < low or (high and value > high):
            raise ValueError(f'{name} is outside its integer window domain')
    if not isinstance(sliding, bool):
        raise ValueError('use_sliding_window must be boolean')
    width = min(length, max(5, round(length * share / 100)))
    step = stride if sliding else width
    return StatisticalWindowPlan(width, tuple(range(0, length - width + 1, step)))
