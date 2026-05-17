from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionProgressPolicy:
    """Central policy for tqdm usage in detection runtime and model heads."""

    enabled: bool = False
    leave: bool = False
    show_postfix: bool = True
    stage_tuning_enabled: bool | None = None
    head_training_enabled: bool | None = None

    def _resolve_scope_enabled(self, scoped_flag: bool | None) -> bool:
        if scoped_flag is None:
            return bool(self.enabled)
        return bool(scoped_flag)

    def is_stage_tuning_enabled(self) -> bool:
        return self._resolve_scope_enabled(self.stage_tuning_enabled)

    def is_head_training_enabled(self) -> bool:
        return self._resolve_scope_enabled(self.head_training_enabled)

    def tqdm_kwargs(self, *, scope: str, desc: str, unit: str) -> dict[str, Any]:
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
        return asdict(self)


def resolve_detection_progress_policy(
        policy: DetectionProgressPolicy | dict[str, Any] | bool | None = None,
        *,
        show_progress: bool | None = True,
) -> DetectionProgressPolicy:
    """Normalize bool, dict or policy input into DetectionProgressPolicy."""
    if isinstance(policy, DetectionProgressPolicy):
        resolved = policy
    elif isinstance(policy, dict):
        resolved = DetectionProgressPolicy(
            enabled=bool(policy.get('enabled', False if show_progress is None else show_progress)),
            leave=bool(policy.get('leave', False)),
            show_postfix=bool(policy.get('show_postfix', True)),
            stage_tuning_enabled=policy.get('stage_tuning_enabled'),
            head_training_enabled=policy.get('head_training_enabled'),
        )
    elif isinstance(policy, bool):
        resolved = DetectionProgressPolicy(enabled=bool(policy))
    else:
        resolved = DetectionProgressPolicy(enabled=False if show_progress is None else bool(show_progress))

    if show_progress is None:
        return resolved
    return DetectionProgressPolicy(
        enabled=bool(show_progress),
        leave=resolved.leave,
        show_postfix=resolved.show_postfix,
        stage_tuning_enabled=resolved.stage_tuning_enabled,
        head_training_enabled=resolved.head_training_enabled,
    )
