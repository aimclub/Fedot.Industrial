from __future__ import annotations

import numpy as np

from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting import (
    okhs_advanced as advanced_module,
)
from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting import (
    okhs_experiment_utils as utils_module,
)


def test_split_trajectories_has_no_overlap():
    trajectories = [np.full((3, 1), fill_value=index, dtype=float) for index in range(5)]
    split = utils_module.split_trajectories(trajectories, holdout_size=2)

    assert split.is_right()
    split_value = split.value
    assert split_value.train_indices == (0, 1, 2)
    assert split_value.test_indices == (3, 4)
    assert set(split_value.train_indices).isdisjoint(split_value.test_indices)


def test_validate_experiment_config_fails_early_for_invalid_segment():
    config = utils_module.ExperimentConfig(
        name="bad",
        q_true=0.8,
        dim=1,
        n_train_traj=5,
        n_steps_train=10,
        T_max_train=1.0,
        initial_segment_length=2,
        n_quad_points=5,
        regularization=1e-6,
        plot_part=1.0,
    )

    validation = utils_module.validate_experiment_config(config)

    assert validation.is_left()
    message = validation.either(lambda value: value, lambda value: value)
    assert "required_mode_budget" in message


def test_evaluate_forecast_uses_only_holdout_suffix():
    test_traj = np.array([[0.0], [10.0], [20.0], [30.0]])
    pred_traj = np.array([[999.0], [888.0], [21.0], [29.0]])
    mse = utils_module.evaluate_forecast(test_traj, pred_traj, initial_segment_length=2)

    expected = float(np.mean((np.array([[20.0], [30.0]]) - np.array([[21.0], [29.0]])) ** 2))
    assert mse == expected


def test_run_experiment_uses_holdout_trajectory_and_excludes_it_from_train(monkeypatch):
    def fake_generate(*args, **kwargs):
        time = np.arange(5, dtype=float)
        trajectories = [
            np.full((5, 1), fill_value=0.0),
            np.full((5, 1), fill_value=1.0),
            np.full((5, 1), fill_value=2.0),
        ]
        return time, trajectories

    def fake_runner(**kwargs):
        assert len(kwargs["train_trajectories"]) == 2
        assert np.all(kwargs["train_trajectories"][0] == 0.0)
        assert np.all(kwargs["train_trajectories"][1] == 1.0)
        assert np.all(kwargs["test_traj"] == 2.0)
        return utils_module.ExperimentResult(
            system_name=kwargs["system_name"],
            mse=0.0,
            train_size=2,
            test_size=1,
            initial_segment_length=kwargs["initial_segment_length"],
            prediction_shape=(5, 1),
        )

    monkeypatch.setattr(advanced_module, "generate_trajectories_pycaputo", fake_generate)
    monkeypatch.setattr(advanced_module, "run_experiment_from_artifacts", fake_runner)

    config = utils_module.ExperimentConfig(
        name="holdout-check",
        q_true=0.8,
        dim=1,
        n_train_traj=2,
        n_steps_train=5,
        T_max_train=1.0,
        initial_segment_length=2,
        n_quad_points=5,
        regularization=1e-6,
        plot_part=1.0,
    )

    result = advanced_module.run_experiment(
        config=config,
        dynamics_func=lambda t, y: y,
        dynamics_args=(),
        kernel=advanced_module.RBFKernel(),
        should_plot=False,
    )

    assert result.train_size == 2
    assert result.test_size == 1
