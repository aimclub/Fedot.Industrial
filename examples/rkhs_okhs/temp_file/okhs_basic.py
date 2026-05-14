import matplotlib.pyplot as plt
import numpy as np

from example_common import RBFKernel, generate_trajectories_pycaputo
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import (
    FractionalDMD,
    FractionalLiouvilleOperator,
    OKHSTransformer,
)


def run_test(
    q_true=0.8,
    lambda_param=-2.0,
    n_train_traj=20,
    n_steps=60,
    T_max=2.0,
    kernel_gamma=1.5,
    n_quad_points=30,
    regularization=1e-6,
    x0_test=None,
        initial_segment_length=10,
):
    print("=== Start Fractional DMD Test (Refactored with Initial Trajectory) ===\n")

    def rhs_linear(t, y, lambda_value):
        return lambda_value * np.array(y)

    time_train, generated_trajectories = generate_trajectories_pycaputo(
        rhs_linear,
        lambda_param,
        q_true=q_true,
        n_trajectories=n_train_traj,
        n_steps=n_steps,
        T_max=T_max,
        dim=2,
        seed=42,
    )

    time_train = time_train[:n_steps]
    train_trajectories = [trajectory[:n_steps] for trajectory in generated_trajectories]

    dt = time_train[1] - time_train[0]
    print(f"Generated {len(train_trajectories)} trajectories.")
    print(f"Time: T_max={T_max}, dt={dt:.4f}, steps={n_steps}")

    kernel = RBFKernel(gamma=kernel_gamma)
    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        n_quad_points=n_quad_points,
        dt=dt,
    )

    print("Fitting OKHS Transformer...")
    okhs.fit(train_trajectories)

    print("Fitting Fractional Liouville Operator...")
    liouville_op = FractionalLiouvilleOperator(
        okhs_transformer=okhs,
        n_quad_points=n_quad_points,
    )
    liouville_op.fit()
    print(f"  Eigenvalues found: {liouville_op.eigenvalues_[:3]} ...")

    print("Fitting Fractional DMD...")
    fdmd = FractionalDMD(
        liouville_operator=liouville_op,
        n_quad_points=n_quad_points,
        regularization=regularization,
    )
    fdmd.fit()

    print("\nModel fitted successfully.")

    if x0_test is None:
        x0_test = np.array([0.5, -0.5])

    _, test_trajectories = generate_trajectories_pycaputo(
        rhs_linear,
        lambda_param,
        q_true=q_true,
        n_trajectories=1,
        n_steps=n_steps,
        T_max=T_max,
        dim=len(x0_test),
        seed=43,
        ic_low=float(np.min(x0_test)),
        ic_high=float(np.max(x0_test)),
    )
    true_traj = test_trajectories[0][:n_steps].copy()
    true_traj[0] = x0_test

    initial_segment = true_traj[:initial_segment_length]
    time = time_train

    n = min(len(time), len(true_traj))
    time = time[:n]
    true_traj = true_traj[:n]

    pred_traj = fdmd.predict(initial_segment, time)[:n]

    forecast_start = min(initial_segment_length, n - 1)
    forecast_true = true_traj[forecast_start:]
    forecast_pred = pred_traj[forecast_start:]
    mse = np.mean((forecast_true - forecast_pred) ** 2)

    print(f"Forecast MSE (from t={time[forecast_start]:.2f}): {mse:.6e}")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time, true_traj[:, 0], "k--", label="Ground Truth", linewidth=2)
    plt.plot(
        time[:initial_segment_length],
        initial_segment[:, 0],
        "bo-",
        label="Initial Segment",
        markersize=4,
        alpha=0.7,
    )
    plt.plot(
        time[: len(true_traj) // 2],
        pred_traj[: len(true_traj) // 2, 0],
        "r-",
        label="Fractional DMD Forecast",
        alpha=0.8,
    )
    plt.axvline(
        x=time[initial_segment_length - 1],
        color="gray",
        linestyle=":",
        label="Forecast Start",
        alpha=0.6,
    )
    plt.title(f"Component 1 (q={q_true})")
    plt.xlabel("Time [s]")
    plt.ylabel("State $x_1$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(time, true_traj[:, 1], "k--", label="Ground Truth", linewidth=2)
    plt.plot(
        time[:initial_segment_length],
        initial_segment[:, 1],
        "bo-",
        label="Initial Segment",
        markersize=4,
        alpha=0.7,
    )
    plt.plot(time, pred_traj[:, 1], "r-", label="Fractional DMD Forecast", alpha=0.8)
    plt.axvline(
        x=time[initial_segment_length - 1],
        color="gray",
        linestyle=":",
        label="Forecast Start",
        alpha=0.6,
    )
    plt.title(f"Component 2 (q={q_true})")
    plt.xlabel("Time [s]")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test(
        q_true=0.7,
        lambda_param=-1.5,
        n_train_traj=10,
        n_steps=1000,
        T_max=1.5,
        kernel_gamma=2.0,
        regularization=1e-7,
        initial_segment_length=50,
    )
