from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from pymonad.either import Either, Left, Right

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import (
    FractionalDMD,
    FractionalLiouvilleOperator,
    OKHSTransformer,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    q_true: float
    dim: int
    n_train_traj: int
    n_steps_train: int
    T_max_train: float
    initial_segment_length: int
    n_quad_points: int
    regularization: float
    plot_part: float
    holdout_size: int = 1
    seed: int = 42
    ic_low: float = -1.0
    ic_high: float = 1.0
    required_mode_budget: int | None = None


@dataclass(frozen=True)
class TrajectorySplit:
    train_trajectories: tuple[np.ndarray, ...]
    test_trajectories: tuple[np.ndarray, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentResult:
    system_name: str
    mse: float
    train_size: int
    test_size: int
    initial_segment_length: int
    prediction_shape: tuple[int, ...]


@dataclass(frozen=True)
class PipelineArtifacts:
    okhs: OKHSTransformer
    liouville_operator: FractionalLiouvilleOperator
    fdmd: FractionalDMD
    dt: float


def _unwrap_either(result: Either) -> Any:
    return result.either(lambda value: value, lambda value: value)


def validate_experiment_config(config: ExperimentConfig) -> Either:
    required_mode_budget = (
        config.required_mode_budget
        if config.required_mode_budget is not None
        else config.n_train_traj
    )

    if config.n_train_traj < 2:
        return Left("n_train_traj must be >= 2")
    if config.holdout_size < 1:
        return Left("holdout_size must be >= 1")
    if config.n_steps_train < config.initial_segment_length:
        return Left("n_steps_train must be >= initial_segment_length")
    if not (0 < config.q_true <= 1):
        return Left("q_true must be in the interval (0, 1]")
    if config.initial_segment_length * config.dim < required_mode_budget:
        return Left(
            "initial_segment_length * dim must be >= required_mode_budget"
        )
    return Right(config)


def split_trajectories(
        trajectories: Sequence[np.ndarray],
        holdout_size: int = 1,
) -> Either:
    total = len(trajectories)
    if holdout_size < 1:
        return Left("holdout_size must be >= 1")
    if total <= holdout_size:
        return Left("number of trajectories must be greater than holdout_size")

    split_index = total - holdout_size
    return Right(
        TrajectorySplit(
            train_trajectories=tuple(trajectories[:split_index]),
            test_trajectories=tuple(trajectories[split_index:]),
            train_indices=tuple(range(split_index)),
            test_indices=tuple(range(split_index, total)),
        )
    )


def fit_okhs_fdmd_pipeline(
        time: np.ndarray,
        train_trajectories: Sequence[np.ndarray],
        q_true: float,
        kernel: Any,
        n_quad_points: int,
        regularization: float,
) -> PipelineArtifacts:
    dt = float(time[1] - time[0])

    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        n_quad_points=n_quad_points,
        dt=dt,
    )
    okhs.fit(list(train_trajectories))

    liouville_operator = FractionalLiouvilleOperator(
        okhs_transformer=okhs,
        n_quad_points=n_quad_points,
    )
    liouville_operator.fit()

    fdmd = FractionalDMD(
        liouville_operator=liouville_operator,
        n_quad_points=n_quad_points,
        regularization=regularization,
    )
    fdmd.fit()

    return PipelineArtifacts(
        okhs=okhs,
        liouville_operator=liouville_operator,
        fdmd=fdmd,
        dt=dt,
    )


def evaluate_forecast(
        test_traj: np.ndarray,
        pred_traj: np.ndarray,
        initial_segment_length: int,
) -> float:
    forecast_true = test_traj[initial_segment_length:]
    forecast_pred = pred_traj[initial_segment_length:]
    return float(np.mean((forecast_true - forecast_pred) ** 2))


def plot_forecast(
        system_name: str,
        time: np.ndarray,
        test_traj: np.ndarray,
        pred_traj: np.ndarray,
        initial_segment_length: int,
        plot_part: float,
        dim: int,
        q_true: float,
) -> None:
    plot_end = int(len(time) * plot_part)
    initial_segment = test_traj[:initial_segment_length]

    if dim == 1:
        plt.figure(figsize=(6, 4))
        plt.plot(time[:plot_end], test_traj[:plot_end, 0], "k--", label="Ground Truth", linewidth=2)
        plt.plot(time[:initial_segment_length], initial_segment[:, 0], "bo-", label="Initial Segment", markersize=4,
                 alpha=0.6)
        plt.plot(time[:plot_end], pred_traj[:plot_end, 0], "r-", label="Forecast", alpha=0.8)
        plt.axvline(x=time[initial_segment_length - 1], color="gray", linestyle=":", label="Start Forecast")
        plt.title(f"{system_name} (q={q_true})")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    plt.figure(figsize=(12, 5))
    for index in range(min(dim, 2)):
        plt.subplot(1, 2, index + 1)
        plt.plot(time[:plot_end], test_traj[:plot_end, index], "k--", label="Ground Truth", linewidth=2)
        plt.plot(time[:initial_segment_length], initial_segment[:, index], "bo-", label="Init", markersize=4, alpha=0.6)
        plt.plot(time[:plot_end], pred_traj[:plot_end, index], "r-", label="Forecast", alpha=0.8)
        plt.axvline(x=time[initial_segment_length - 1], color="gray", linestyle=":")
        plt.title(f"{system_name}: x_{index + 1}")
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)
        if index == 0:
            plt.ylabel("State")
            plt.legend()
    plt.tight_layout()
    plt.show()


def run_experiment_from_artifacts(
        system_name: str,
        time: np.ndarray,
        train_trajectories: Sequence[np.ndarray],
        test_traj: np.ndarray,
        q_true: float,
        dim: int,
        kernel: Any,
        n_quad_points: int,
        regularization: float,
        initial_segment_length: int,
        plot_part: float,
        should_plot: bool = True,
) -> ExperimentResult:
    artifacts = fit_okhs_fdmd_pipeline(
        time=time,
        train_trajectories=train_trajectories,
        q_true=q_true,
        kernel=kernel,
        n_quad_points=n_quad_points,
        regularization=regularization,
    )
    initial_segment = test_traj[:initial_segment_length]
    pred_traj = artifacts.fdmd.predict(initial_segment, time)
    mse = evaluate_forecast(test_traj, pred_traj, initial_segment_length)

    if should_plot:
        plot_forecast(
            system_name=system_name,
            time=time,
            test_traj=test_traj,
            pred_traj=pred_traj,
            initial_segment_length=initial_segment_length,
            plot_part=plot_part,
            dim=dim,
            q_true=q_true,
        )

    return ExperimentResult(
        system_name=system_name,
        mse=mse,
        train_size=len(train_trajectories),
        test_size=1,
        initial_segment_length=initial_segment_length,
        prediction_shape=tuple(pred_traj.shape),
    )


def ensure_valid_config(config: ExperimentConfig) -> ExperimentConfig:
    validation = validate_experiment_config(config)
    if validation.is_left():
        raise ValueError(_unwrap_either(validation))
    return config
