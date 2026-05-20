from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedot_ind.core.models.ts_forecasting.progress_policy import BaseProgressPolicy


@dataclass(frozen=True)
class DetectionProgressPolicy(BaseProgressPolicy):
    """Detection-specific progress policy.

    The class currently shares all behavior with :class:`BaseProgressPolicy`,
    but remains a dedicated type for:
    - clearer API contracts in detection modules;
    - future detection-only scopes (e.g., calibration diagnostics).
    """


def resolve_detection_progress_policy(
        policy: DetectionProgressPolicy | dict[str, Any] | bool | None = None,
        *,
        show_progress: bool | None = True,
) -> DetectionProgressPolicy:
    """Normalize user config into :class:`DetectionProgressPolicy`.

    Parameters
    ----------
    policy:
        Policy in one of supported forms:
        - ready dataclass instance,
        - plain dict from config/JSON,
        - bool shortcut (on/off),
        - None (use defaults).
    show_progress:
        Optional hard override for the global ``enabled`` flag.
    """
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
