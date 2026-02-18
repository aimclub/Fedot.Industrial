from fedot_ind.core.operation.transformation.representation.kernel.occupation import OccupationKernelFunctional
import numpy as np
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.okhs import OKHSTransformer, FractionalLiouvilleOperator, FractionalDMD
import matplotlib.pyplot as plt
from scipy.special import gamma
from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode import caputo
from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve



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
    
def mittag_leffler_scalar(z, alpha):
    """
    Скалярная реализация функции Миттаг-Леффлера E_{alpha, 1}(z)
    для генерации эталонных данных (Ground Truth).
    """
    res = 0
    for k in range(200):
        term = (z ** k) / gamma(alpha * k + 1)
        res += term
        if np.abs(term) < 1e-15:
            break
    return np.real(res)

def generate_trajectories_ml(q_true, lambda_param, n_train_traj, n_steps, T_max, dim=2, seed=42):
    """
    Генерация набора траекторий для системы
    D^{q_true} x(t) = lambda_param * x(t)
    с аналитическим решением через функцию Миттаг-Леффлера.
    """
    time = np.linspace(0, T_max, n_steps)
    train_trajectories = []

    np.random.seed(seed)
    initial_conditions = [np.random.uniform(-1, 1, dim) for _ in range(n_train_traj)]
    for x0 in initial_conditions:
        traj = np.zeros((n_steps, dim))
        for t_idx, t in enumerate(time):
            z = lambda_param * (t ** q_true)
            ml_val = mittag_leffler_scalar(z, q_true)
            traj[t_idx, :] = x0 * ml_val
        train_trajectories.append(traj)

    return time, train_trajectories

def generate_trajectories_pycaputo(
    q_true, lambda_param, n_train_traj, n_steps, T_max, dim=2, seed=42
):
    """
    Генерация набора траекторий для системы D^{q_true} x(t) = lambda_param * x(t)
    """
    t0 = 0.0
    dt = T_max / (n_steps - 1)

    rng = np.random.default_rng(seed)
    initial_conditions = rng.uniform(-1.0, 1.0, size=(n_train_traj, dim))

    def rhs(t, y):
        return lambda_param * y

    train_trajectories = []

    for x0 in initial_conditions:
        ds = tuple(D(q_true) for _ in range(dim))
        stepper = caputo.PECE(
            ds=ds,
            control=make_fixed_controller(dt, tstart=t0, tfinal=T_max),
            source=rhs,
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

def run_test(
    time,
    train_trajectories,
    q_true=0.8,
    lambda_param=-2.0,
    n_train_traj=15,
    n_steps=50,
    kernel_gamma=0.5,
    integration_method='gaussian',
    n_quad_points=40,
    n_gaussian_points=40,
    x0_test=None,
):
    print("=== Start Fractional DMD Test ===\n")
    print(f"Generated {n_train_traj} training trajectories.")

    # --- Обучение модели ---
    kernel = RBFKernel(gamma=kernel_gamma)

    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        integration_method=integration_method,
        n_quad_points=n_quad_points
    )
    okhs.fit(train_trajectories)

    liouville_op = FractionalLiouvilleOperator(okhs, n_gaussian_points=n_gaussian_points)
    liouville_op.fit()

    fdmd = FractionalDMD(liouville_op)
    fdmd.fit()

    print("\nModel fitted successfully.")
    
    if x0_test is None:
        x0_test = np.array([0.8, -0.5])
    
    true_traj = np.zeros((n_steps, 2))
    for t_idx, t in enumerate(time):
        z = lambda_param * (t ** q_true)
        ml_val = mittag_leffler_scalar(z, q_true)
        true_traj[t_idx, :] = x0_test * ml_val

    pred_traj = fdmd.predict(x0_test, time)


    # --- Визуализация ---
    plt.figure(figsize=(12, 5))
    print(f"Predicted start point: ({true_traj[0, 0]}, {true_traj[0, 1]})")


    # График координаты X1
    plt.subplot(1, 2, 1)
    plt.plot(time, true_traj[:, 0], 'k--', label='Analytical (True)', linewidth=2)
    plt.plot(time, pred_traj[:, 0], 'r-', label='Fractional DMD (Pred)')
    plt.scatter([0], [x0_test[0]], color='blue', label='x0')
    plt.title(f"Feature 1 (q={q_true})")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend()
    plt.grid(True, alpha=0.3)


    # График координаты X2
    plt.subplot(1, 2, 2)
    plt.plot(time, true_traj[:, 1], 'k--', label='Analytical (True)', linewidth=2)
    plt.plot(time, pred_traj[:, 1], 'r-', label='Fractional DMD (Pred)')
    plt.scatter([0], [x0_test[1]], color='blue', label='x0')
    plt.title(f"Feature 2 (q={q_true})")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()


    # Оценка ошибки
    mse = np.mean((true_traj - pred_traj) ** 2)
    print(f"\nPrediction MSE: {mse:.6e}")



if __name__ == "__main__":
    q_true=0.8
    lambda_param=-2.0
    n_train_traj=25
    n_steps=50
    T_max=1.0

    #Хочется протестировать обе генерации данных - и через ML, и через PyCaputo и сравнинть на них реультаты предсказаний. 
    time, train_trajectories = generate_trajectories_ml(
        q_true=q_true,
        lambda_param=lambda_param,
        n_train_traj=n_train_traj,
        n_steps=n_steps,
        T_max=T_max,
    )
    run_test(
        time=time,
        train_trajectories=train_trajectories,
    )

    time, train_trajectories = generate_trajectories_pycaputo(
        q_true=q_true,
        lambda_param=lambda_param,
        n_train_traj=n_train_traj,
        n_steps=n_steps,
        T_max=T_max,
    )

    run_test(
        time=time,
        train_trajectories=train_trajectories,
        q_true=0.8,
        lambda_param=-2.0,
        n_train_traj=25,
        n_steps=50,
        T_max=1.0,
        kernel_gamma=0.5,
        integration_method='gaussian',
        n_quad_points=40,
        n_gaussian_points=40,
        x0_test=None,
        mse_threshold=1e-2,
            )

