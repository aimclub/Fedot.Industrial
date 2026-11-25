import pytest

from industrial_forecasting.progress import ForecastingProgressPolicy, resolve_forecasting_progress_policy


@pytest.mark.parametrize('policy', [False, {'enabled': False}, ForecastingProgressPolicy(enabled=False)])
def test_explicitly_disabled_progress_is_kept_without_an_override(policy):
    assert resolve_forecasting_progress_policy(policy).enabled is False
    assert resolve_forecasting_progress_policy(policy, show_progress=True).enabled is True


def test_progress_defaults_to_off_but_preserves_individual_scopes():
    assert resolve_forecasting_progress_policy().enabled is False
    scoped = ForecastingProgressPolicy(enabled=True, leave=True, stage_tuning_enabled=True)
    resolved = resolve_forecasting_progress_policy(scoped, show_progress=False)
    assert not resolved.is_head_training_enabled()
    assert resolved.is_stage_tuning_enabled()
    assert resolved.leave
