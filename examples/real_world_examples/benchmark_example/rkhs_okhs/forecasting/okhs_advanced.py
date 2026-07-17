from __future__ import annotations

import numpy as np
import torch

from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.example_common import (
    RBFKernel,
    generate_trajectories_pycaputo,
)
from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.okhs_experiment_utils import (
    ExperimentConfig,
    ExperimentResult,
    ensure_valid_config,
    run_experiment_from_artifacts,
    split_trajectories,
)


def rhs_linear(t, y, lambda_param):
    return lambda_param * np.array(y)


def rhs_logistic(t, y, r):
    return r * y * (1.0 - y)


def rhs_quadratic(t, y, a, b):
    return a * y - b * y**2


def rhs_mu_cubic(t, y, mu):
    return mu * (1.0 - y**2) * y - y


def run_experiment(
    config: ExperimentConfig,
    dynamics_func,
    dynamics_args,
    kernel,
    should_plot: bool = False,
) -> ExperimentResult:
    config = ensure_valid_config(config)
    time, trajectories = generate_trajectories_pycaputo(
        dynamics_func,
        *dynamics_args,
        q_true=config.q_true,
        n_trajectories=config.n_train_traj + config.holdout_size,
        n_steps=config.n_steps_train,
        T_max=config.T_max_train,
        dim=config.dim,
        seed=config.seed,
        ic_low=config.ic_low,
        ic_high=config.ic_high,
    )
    split = split_trajectories(trajectories, holdout_size=config.holdout_size)
    if split.is_left():
        raise ValueError(split.either(lambda value: value, lambda value: value))
    split_value = split.value
    return run_experiment_from_artifacts(
        system_name=config.name,
        time=time,
        train_trajectories=split_value.train_trajectories,
        test_traj=split_value.test_trajectories[0],
        q_true=config.q_true,
        dim=config.dim,
        kernel=kernel,
        n_quad_points=config.n_quad_points,
        regularization=config.regularization,
        initial_segment_length=config.initial_segment_length,
        plot_part=config.plot_part,
        should_plot=should_plot,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def run_test(
    system_name,
    time,
    train_trajectories,
    test_traj,
    q_true,
    dim,
    kernel=RBFKernel(gamma=0.5),
    n_quad_points=100,
    regularization=1e-3,
    initial_segment_length=10,
    plot_part=1.0,
):
    return run_experiment_from_artifacts(
        system_name=system_name,
        time=time,
        train_trajectories=train_trajectories,
        test_traj=test_traj,
        q_true=q_true,
        dim=dim,
        kernel=kernel,
        n_quad_points=n_quad_points,
        regularization=regularization,
        initial_segment_length=initial_segment_length,
        plot_part=plot_part,
        should_plot=False,
    )


__all__ = [
    "RBFKernel",
    "rhs_linear",
    "rhs_logistic",
    "rhs_mu_cubic",
    "rhs_quadratic",
    "run_experiment",
    "run_test",
]
