from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BaseProgressPolicy:
    """Shared tqdm policy for runtime flows with scoped progress bars.

    The benchmark/runtime code has two independent places where tqdm bars may
    appear:
    - global orchestration level (dataset/model/series loops)
    - scoped model internals (stage tuning, neural head training)

    This base class centralizes all switch logic so task-specific policies
    (forecasting, detection) only need type-level specialization.
    """

    enabled: bool = False
    leave: bool = False
    show_postfix: bool = True
    stage_tuning_enabled: bool | None = None
    head_training_enabled: bool | None = None

    def _resolve_scope_enabled(self, scoped_flag: bool | None) -> bool:
        """Resolve per-scope enable flag with fallback to global ``enabled``."""
        if scoped_flag is None:
            return bool(self.enabled)
        return bool(scoped_flag)

    def is_stage_tuning_enabled(self) -> bool:
        """Return whether stage-tuning tqdm bars should be visible."""
        return self._resolve_scope_enabled(self.stage_tuning_enabled)

    def is_head_training_enabled(self) -> bool:
        """Return whether head-training tqdm bars should be visible."""
        return self._resolve_scope_enabled(self.head_training_enabled)

    def tqdm_kwargs(self, *, scope: str, desc: str, unit: str) -> dict[str, Any]:
        """Build normalized kwargs for ``tqdm``.

        Parameters
        ----------
        scope:
            Logical scope name (`stage_tuning`, `head_training`, or custom).
        desc:
            Human-readable progress label.
        unit:
            Iteration unit label shown by tqdm.
        """
        if scope == 'stage_tuning':
            enabled = self.is_stage_tuning_enabled()
        elif scope == 'head_training':
            enabled = self.is_head_training_enabled()
        else:
            enabled = bool(self.enabled)
        return {
            'desc': desc,
            'unit': unit,
            'leave': bool(self.leave),
            'disable': not enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy into JSON-friendly dict for metadata artifacts."""
        return asdict(self)


@dataclass(frozen=True)
class ForecastingProgressPolicy(BaseProgressPolicy):
    """Forecasting-specific alias of :class:`BaseProgressPolicy`.

    Kept as an explicit type for public API readability and typing compatibility
    in forecasting modules.
    """


def resolve_forecasting_progress_policy(
        policy: ForecastingProgressPolicy | dict[str, Any] | bool | None = None,
        *,
        show_progress: bool | None = True,
) -> ForecastingProgressPolicy:
    """Normalize bool, dict or policy input into ForecastingProgressPolicy."""
    if isinstance(policy, ForecastingProgressPolicy):
        resolved = policy
    elif isinstance(policy, dict):
        resolved = ForecastingProgressPolicy(
            enabled=bool(policy.get('enabled', False if show_progress is None else show_progress)),
            leave=bool(policy.get('leave', False)),
            show_postfix=bool(policy.get('show_postfix', True)),
            stage_tuning_enabled=policy.get('stage_tuning_enabled'),
            head_training_enabled=policy.get('head_training_enabled'),
        )
    elif isinstance(policy, bool):
        resolved = ForecastingProgressPolicy(enabled=bool(policy))
    else:
        resolved = ForecastingProgressPolicy(enabled=False if show_progress is None else bool(show_progress))

    if show_progress is None:
        return resolved
    return ForecastingProgressPolicy(
        enabled=bool(show_progress),
        leave=resolved.leave,
        show_postfix=resolved.show_postfix,
        stage_tuning_enabled=resolved.stage_tuning_enabled,
        head_training_enabled=resolved.head_training_enabled,
    )
