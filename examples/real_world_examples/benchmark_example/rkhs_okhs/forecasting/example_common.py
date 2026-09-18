from __future__ import annotations

import numpy as np
import torch

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


class MittagLefflerKernel:
    def __init__(self, q: float = 0.8, beta: float = 1.0, gamma: float = 1.0):
        if not 0.0 < float(q) <= 1.0:
            raise ValueError("q must be in the interval (0, 1].")
        self.q = float(q)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def _compute_batch_kernel(self, x, y):
        x_tensor = torch.as_tensor(x)
        y_tensor = torch.as_tensor(y)
        dist_sq = torch.sum((x_tensor - y_tensor) ** 2, dim=-1)
        values = self._evaluate(-self.gamma * dist_sq.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=x_tensor.dtype, device=x_tensor.device)

    def _compute_single_kernel(self, x, y):
        x_tensor = torch.as_tensor(x)
        y_tensor = torch.as_tensor(y)
        dist_sq = torch.sum((x_tensor - y_tensor) ** 2)
        return float(np.asarray(self._evaluate(-self.gamma * float(dist_sq))).reshape(-1)[0])

    def _evaluate(self, argument):
        return self.beta * mittag_leffler(np.asarray(argument, dtype=float), self.q, 1.0)


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
        derivatives = tuple(D(q_true) for _ in range(dim))
        stepper = caputo.PECE(
            ds=derivatives,
            control=make_fixed_controller(dt, tstart=t0, tfinal=T_max),
            source=source_func,
            y0=(x0,),
            corrector_iterations=1,
        )
        values = [x0]
        for event in evolve(stepper):
            if isinstance(event, StepCompleted):
                values.append(event.y)
        trajectory = np.array(values, dtype=float).reshape(-1, dim)
        if len(trajectory) > n_steps + 1:
            trajectory = trajectory[: n_steps + 1]
        elif len(trajectory) < n_steps + 1:
            padding = np.tile(trajectory[-1], (n_steps + 1 - len(trajectory), 1))
            trajectory = np.vstack((trajectory, padding))
        trajectories.append(trajectory)

    time = np.linspace(0, T_max, n_steps + 1)
    return time, trajectories


__all__ = ["MittagLefflerKernel", "RBFKernel", "generate_trajectories_pycaputo"]
