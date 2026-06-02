from __future__ import annotations

import pytest
from pymonad.either import Left, Right

from fedot_ind.api.flow import branch, named_step, pipe, tap, to_either, unwrap_or_raise


def test_tap_returns_original_value_after_effect():
    calls = []
    payload = {"value": 1}

    result = pipe(payload, tap(calls.append), lambda value: value["value"] + 1)

    assert unwrap_or_raise(result) == 2
    assert calls == [payload]


def test_branch_chooses_expected_side_deterministically():
    step = branch(
        predicate=lambda value: value > 0,
        left=lambda value: f"left:{value}",
        right=lambda value: f"right:{value}",
    )

    assert unwrap_or_raise(pipe(2, step)) == "right:2"
    assert unwrap_or_raise(pipe(0, step)) == "left:0"


def test_unwrap_or_raise_preserves_right_value():
    assert unwrap_or_raise(Right({"ok": True})) == {"ok": True}


def test_unwrap_or_raise_maps_left_error_to_exception():
    with pytest.raises(RuntimeError, match="bad config"):
        unwrap_or_raise(Left("bad config"), error_mapper=lambda error: RuntimeError(str(error)))


def test_to_either_returns_left_for_failed_predicate():
    result = to_either("classification", lambda value: value == "regression", "invalid task")

    assert result.is_left()
    with pytest.raises(ValueError, match="invalid task"):
        unwrap_or_raise(result)


def test_pipe_accepts_existing_left_and_skips_steps():
    calls = []

    result = pipe(Left("stop"), tap(calls.append), lambda value: value + 1)

    assert result.is_left()
    assert calls == []


def test_named_step_exposes_stable_debug_name():
    step = named_step("increment", lambda value: value + 1)

    assert step.__name__ == "increment"
    assert step.step_name == "increment"
    assert unwrap_or_raise(pipe(1, step)) == 2
