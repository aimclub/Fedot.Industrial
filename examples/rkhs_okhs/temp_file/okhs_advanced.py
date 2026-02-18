import numpy as np
import matplotlib.pyplot as plt

from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode import caputo
from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import (
    OKHSTransformer,
    FractionalLiouvilleOperator,
    FractionalDMD,
)

class RBFKernel:
    """
    Радиально-базисное ядро (Gaussian Kernel):
    K(x, y) = exp(-gamma * ||x - y||^2)
    """
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _compute_single_kernel(self, x, y):
        dist_sq = np.sum((x - y) ** 2)
        return np.exp(-self.gamma * dist_sq)


def generate_trajectories_pycaputo(
    f, *f_args,
    q_true,
    n_train_traj,
    n_steps,
    T_max,
    dim=2,
    seed=42,
    ic_low=-1.0,
    ic_high=1.0,
):
    """
    Генерация набора траекторий для системы D^{q_true} y(t) = f(t, y, *f_args)
    """
    t0 = 0.0
    dt = T_max / (n_steps - 1)

    rng = np.random.default_rng(seed)
    initial_conditions = rng.uniform(ic_low, ic_high, size=(n_train_traj, dim))

    train_trajectories = []

    for x0 in initial_conditions:
        ds = tuple(D(q_true) for _ in range(dim))
        stepper = caputo.PECE(
            ds=ds,
            control=make_fixed_controller(dt, tstart=t0, tfinal=T_max),
            source=lambda t, y: f(t, y, *f_args),
            y0=(x0,),
            corrector_iterations=1,
        )

        ts = []
        ys = []

        for event in evolve(stepper):
            assert isinstance(event, StepCompleted)
            ts.append(event.t)
            ys.append(event.y)

        traj = np.array(ys, dtype=float)
        train_trajectories.append(traj)

    time = np.array(ts, dtype=float)
    return time, train_trajectories


# Генерация эталонной траектории (для валидации)
def generate_reference_trajectory(
    f, *f_args,
    q_true,
    x0,
    n_steps,
    T_max,
    dim=2,
):
    """
    Та же процедура для одной траектории.
    """
    t0 = 0.0
    dt = T_max / (n_steps - 1)

    ds = tuple(D(q_true) for _ in range(dim))
    stepper = caputo.PECE(
        ds=ds,
        control=make_fixed_controller(dt, tstart=t0, tfinal=T_max),
        source=lambda t, y: f(t, y, *f_args),
        y0=(x0,),
        corrector_iterations=1,
    )

    ts = []
    ys = []

    for event in evolve(stepper):
        assert isinstance(event, StepCompleted)
        ts.append(event.t)
        ys.append(event.y)

    time = np.array(ts, dtype=float)
    traj = np.array(ys, dtype=float)
    return time, traj


def run_test(
    *,
    system_name,
    f, f_args,
    time,
    train_trajectories,
    q_true,
    n_steps,
    T_max,
    dim,
    kernel_gamma=0.5,
    integration_method="gaussian",
    n_quad_points=40,
    n_gaussian_points=40,
    x0_test=None,
):
    print(f"\n=== Fractional DMD Test: {system_name} ===")
    print(f"Training trajectories: {len(train_trajectories)}")

    if x0_test is None:
        x0_test = np.full(dim, 0.2, dtype=float)
    else:
        x0_test = np.asarray(x0_test, dtype=float)

    # --- Fit ---
    kernel = RBFKernel(gamma=kernel_gamma)

    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        integration_method=integration_method,
        n_quad_points=n_quad_points,
    )
    okhs.fit(train_trajectories)

    liouville_op = FractionalLiouvilleOperator(okhs, n_gaussian_points=n_gaussian_points)
    liouville_op.fit()

    fdmd = FractionalDMD(liouville_op)
    fdmd.fit()

    print("Model fitted successfully.")

    _, true_traj = generate_reference_trajectory(
        f, *f_args,
        q_true=q_true,
        x0=x0_test,
        n_steps=n_steps,
        T_max=T_max,
        dim=dim,
    )

    # --- Prediction ---
    pred_traj = fdmd.predict(x0_test, time)

    # --- Plots ---
    if dim == 1:
        plt.figure(figsize=(6, 4))
        plt.plot(time, true_traj[:, 0], "k--", label="Reference", linewidth=2)
        plt.plot(time, pred_traj[:, 0], "r-", label="Fractional DMD")
        plt.scatter([0], [x0_test[0]], color="blue", label="x0")
        plt.title(f"{system_name} (q={q_true})")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(12, 5))
        for j in range(min(dim, 2)):
            plt.subplot(1, 2, j + 1)
            plt.plot(time, true_traj[:, j], "k--", label="Reference", linewidth=2)
            plt.plot(time, pred_traj[:, j], "r-", label="Fractional DMD")
            plt.scatter([0], [x0_test[j]], color="blue", label="x0")
            plt.title(f"{system_name}: feature {j+1} (q={q_true})")
            plt.xlabel("Time")
            plt.grid(True, alpha=0.3)
            if j == 0:
                plt.ylabel("State")
            plt.legend()
        plt.tight_layout()
        plt.show()

    mse = float(np.mean((true_traj - pred_traj) ** 2))
    print(f"Prediction MSE: {mse:.6e}")
    return mse


def rhs(t, y, lambda_param):
    return lambda_param * y

def rhs_logistic(t, y, r):
    """D^α y = r * y * (1 - y)"""
    return r * y * (1.0 - y)

def rhs_quadratic(t, y, a, b):
    """D^α y = a * y - b * y^2"""
    return a * y - b * y**2


def rhs_mu_cubic(t, y, mu):
    """D^α y = mu * (1 - y^2) * y - y"""
    return mu * (1.0 - y**2) * y - y


if __name__ == "__main__":
    q_true = 0.8
    n_train_traj = 25
    n_steps = 50
    T_max = 1.0
    T_test = 0.2
    cases = [
        dict(
            name="Linear: D^α y = λ*y",
            f=rhs,
            args=(-2.0,),
            dim=2,
            ic_low=-1.0,
            ic_high=1.0,
            x0_test=np.array([0.5, -0.5]),
        ),
        dict(
            name="Logistic: D^α y = r*y*(1-y)",
            f=rhs_logistic,
            args=(2.0,),    # если 2, то работает, если 4, то уже нет - слишком быстро растет
            dim=1,
            ic_low=0.1,
            ic_high=0.9,
            x0_test=np.array([0.3]),
        ),
        dict(
            name="Quadratic: D^α y = a*y - b*y²",
            f=rhs_quadratic,
            args=(2.0, 1.0),
            dim=1,
            ic_low=0.1,
            ic_high=0.9,
            x0_test=np.array([0.4]),
        ),
        dict(
            name="Van der Pol-like: D^α y = μ*(1-y²)*y - y",
            f=rhs_mu_cubic,
            args=(3.0,),
            dim=1,
            ic_low=-0.8,
            ic_high=0.8,
            x0_test=np.array([0.5]),
        ),
    ]

    for cfg in cases:
        time, train_trajectories = generate_trajectories_pycaputo(
            cfg["f"], *cfg["args"],
            q_true=q_true,
            n_train_traj=n_train_traj,
            n_steps=n_steps,
            T_max=T_max,
            dim=cfg["dim"],
            seed=42,
            ic_low=cfg["ic_low"],
            ic_high=cfg["ic_high"],
        )

        time_test = np.linspace(0, T_test, n_steps)
        run_test(
            system_name=cfg["name"],
            f=cfg["f"],
            f_args=cfg["args"],
            time=time_test,
            train_trajectories=train_trajectories,
            q_true=q_true,
            n_steps=n_steps,
            T_max=T_max,
            dim=cfg["dim"],
            kernel_gamma=0.5,
            integration_method="gaussian",
            n_quad_points=40,
            n_gaussian_points=40,
            x0_test=cfg["x0_test"],
        )
