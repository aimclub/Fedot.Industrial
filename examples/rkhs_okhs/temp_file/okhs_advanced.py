import numpy as np

from examples.rkhs_okhs.example_common import (
    MittagLefflerKernel,
    RBFKernel,
    generate_trajectories_pycaputo,
)
from examples.rkhs_okhs.temp_file.okhs_experiment_utils import (
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


def run_experiment(config: ExperimentConfig, dynamics_func, dynamics_args, kernel,
                   should_plot: bool = True) -> ExperimentResult:
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
    """
    Backward-compatible thin wrapper around the refactored experiment utilities.
    """
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
        should_plot=True,
    )


if __name__ == "__main__":
    cases = [
        dict(
            config=ExperimentConfig(
                name="Linear: D^О± y = О»*y",
                q_true=0.8,
                dim=2,
                n_train_traj=100,
                n_steps_train=500,
                T_max_train=5.0,
                initial_segment_length=105,
                n_quad_points=30,
                regularization=1e-7,
                plot_part=1.0,
                seed=42,
                ic_low=-1.0,
                ic_high=1.0,
            ),
            dynamics_func=rhs_linear,
            dynamics_args=(-2.0,),
            kernel=RBFKernel(gamma=1.5),
        ),
        dict(
            config=ExperimentConfig(
                name="Logistic: D^О± y = r*y*(1-y)",
                q_true=0.8,
                dim=1,
                n_train_traj=150,
                n_steps_train=500,
                T_max_train=3.0,
                initial_segment_length=155,
                n_quad_points=30,
                regularization=1e-7,
                plot_part=1.0,
                seed=42,
                ic_low=0.1,
                ic_high=0.9,
            ),
            dynamics_func=rhs_logistic,
            dynamics_args=(1.5,),
            kernel=RBFKernel(gamma=2.0),
        ),
        dict(
            config=ExperimentConfig(
                name="Quadratic: D^О± y = a*y - b*yВІ",
                q_true=0.8,
                dim=1,
                n_train_traj=250,
                n_steps_train=700,
                T_max_train=5.0,
                initial_segment_length=255,
                n_quad_points=30,
                regularization=1e-7,
                plot_part=1.0,
                seed=42,
                ic_low=0.1,
                ic_high=2.0,
            ),
            dynamics_func=rhs_quadratic,
            dynamics_args=(2.0, 1.0),
            kernel=RBFKernel(gamma=3.0),
        ),
    ]

    for case in cases:
        result = run_experiment(
            config=case["config"],
            dynamics_func=case["dynamics_func"],
            dynamics_args=case["dynamics_args"],
            kernel=case["kernel"],
            should_plot=True,
        )
        print(result)
