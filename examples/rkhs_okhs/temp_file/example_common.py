from __future__ import annotations
import torch
import numpy as np

try:
    from pymittagleffler import mittag_leffler
except ImportError:  # pragma: no cover - fallback for local environments without pymittagleffler
    from fedot_ind.core.operation.transformation.representation.kernel.utils import mittag_leffler

try:
    from pycaputo.controller import make_fixed_controller
    from pycaputo.derivatives import CaputoDerivative as D
    from pycaputo.events import StepCompleted
    from pycaputo.fode import caputo
    from pycaputo.stepping import evolve
except ImportError:  # pragma: no cover - optional runtime dependency for examples
    make_fixed_controller = None
    D = None
    StepCompleted = None
    caputo = None
    evolve = None

class RBFKernel:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def _compute_batch_kernel(self, x, y):
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    def _compute_single_kernel(self, x, y):
        x_tensor = torch.as_tensor(x)
        y_tensor = torch.as_tensor(y)
        dist_sq = torch.sum((x_tensor - y_tensor) ** 2)
        
        return torch.exp(-self.gamma * dist_sq).item()

def generate_trajectories_pycaputo(
        f,
        *f_args,
        q_true,
        n_trajectories,
        n_steps,
        T_max,
        dim=2,
        seed=42,
        ic_low=-1.0,
        ic_high=1.0,
):
    if make_fixed_controller is None or D is None or StepCompleted is None or caputo is None or evolve is None:
        raise ImportError("pycaputo is required to generate trajectories for this example.")

    t0 = 0.0
    dt = T_max / (n_steps - 1)

    rng = np.random.default_rng(seed)
    initial_conditions = rng.uniform(ic_low, ic_high, size=(n_trajectories, dim))

    trajectories = []

    def source_func(t, y):
        return np.array(f(t, y, *f_args))

    for x0 in initial_conditions:
        ds = tuple(D(q_true) for _ in range(dim))
        stepper = caputo.PECE(
            ds=ds,
            control=make_fixed_controller(dt, tstart=t0, tfinal=T_max),
            source=source_func,
            y0=(x0,),
            corrector_iterations=1,
        )

        ys = [x0]
        for event in evolve(stepper):
            if isinstance(event, StepCompleted):
                ys.append(event.y)

        traj = np.array(ys, dtype=float).reshape(-1, dim)
        if len(traj) > n_steps + 1:
            traj = traj[:n_steps + 1]
        elif len(traj) < n_steps + 1:
            padding = np.tile(traj[-1], (n_steps + 1 - len(traj), 1))
            traj = np.vstack((traj, padding))

        trajectories.append(traj)

    time = np.linspace(0, T_max, n_steps + 1)
    return time, trajectories



def rhs_linear(t, y, lambda_param):
    return lambda_param * np.array(y)

def rhs_logistic(t, y, r):
    return r * y * (1.0 - y)

def rhs_quadratic(t, y, a, b):
    return a * y - b * y**2

def rhs_mu_cubic(t, y, mu):
    return mu * (1.0 - y**2) * y - y

def rhs_lotka_volterra(t, y, alpha, beta, delta, gamma):
    """
    Дробная модель Лотки-Вольтерры.
    D^q u = alpha*u - beta*u*v
    D^q v = delta*u*v - gamma*v
    """
    u, v = y[0], y[1]
    return np.array([
        alpha * u - beta * u * v,
        delta * u * v - gamma * v
    ])

