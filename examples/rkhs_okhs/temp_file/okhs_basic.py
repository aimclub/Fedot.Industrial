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
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def _compute_single_kernel(self, x, y):
        diff = x - y
        dist_sq = np.dot(diff, diff)
        return np.exp(-self.gamma * dist_sq)


def generate_trajectories_pycaputo(
    q_true, lambda_param, n_train_traj, n_steps, T_max, dim=2, seed=42
):
    """
    Генерация набора траекторий для D^q x = lambda * x.
    """
    dt = T_max / (n_steps - 1)
    t_eval = np.linspace(0, T_max, n_steps)
    
    rng = np.random.default_rng(seed)
    initial_conditions = rng.uniform(-1.0, 1.0, size=(n_train_traj, dim))
    
    train_trajectories = []

    def rhs(t, y):
        return lambda_param * np.array(y)

    for x0 in initial_conditions:
        ds = tuple(D(q_true) for _ in range(dim))
        
        stepper = caputo.PECE(
            ds=ds,
            control=make_fixed_controller(dt, tstart=0.0, tfinal=T_max),
            source=rhs,
            y0=(x0,), 
            corrector_iterations=1,
        )

        ts = []
        ys = []

        for event in evolve(stepper):
            if isinstance(event, StepCompleted):
                ts.append(event.t)
                ys.append(event.y)

        traj = np.array(ys, dtype=float).reshape(-1, dim)
        
        if len(traj) > n_steps:
            traj = traj[:n_steps]
        
        train_trajectories.append(traj)

    return t_eval, train_trajectories


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
    initial_segment_length=10  # Длина начального сегмента для предсказания
):
    print("=== Start Fractional DMD Test (Refactored with Initial Trajectory) ===\n")
    
    # 1. Генерация данных
    time_train, train_trajectories = generate_trajectories_pycaputo(
        q_true=q_true, 
        lambda_param=lambda_param, 
        n_train_traj=n_train_traj, 
        n_steps=n_steps, 
        T_max=T_max
    )
    
    dt = time_train[1] - time_train[0]
    print(f"Generated {len(train_trajectories)} trajectories.")
    print(f"Time: T_max={T_max}, dt={dt:.4f}, steps={n_steps}")

    # 2. Инициализация Pipeline
    kernel = RBFKernel(gamma=kernel_gamma)

    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        n_quad_points=n_quad_points,
        dt=dt
    )
    
    print("Fitting OKHS Transformer...")
    okhs.fit(train_trajectories)

    print("Fitting Fractional Liouville Operator...")
    liouville_op = FractionalLiouvilleOperator(
        okhs_transformer=okhs, 
        n_quad_points=n_quad_points
    )
    liouville_op.fit()
    
    print(f"  Eigenvalues found: {liouville_op.eigenvalues_[:3]} ...")

    print("Fitting Fractional DMD...")
    fdmd = FractionalDMD(
        liouville_operator=liouville_op,
        n_quad_points=n_quad_points,
        regularization=regularization
    )
    fdmd.fit()

    print("\nModel fitted successfully.")
    
    # 3. Генерация тестовой траектории (ground truth)
    if x0_test is None:
        x0_test = np.array([0.5, -0.5])
    
    # Генерируем полную траекторию для ground truth
    dim = len(x0_test)
    
    def rhs_test(t, y):
        return lambda_param * np.array(y)
    
    ds_test = tuple(D(q_true) for _ in range(dim))
    stepper_test = caputo.PECE(
        ds=ds_test,
        control=make_fixed_controller(dt, tstart=0.0, tfinal=T_max),
        source=rhs_test,
        y0=(x0_test,),
        corrector_iterations=1,
    )
    
    ts_true = []
    ys_true = []
    
    for event in evolve(stepper_test):
        if isinstance(event, StepCompleted):
            ts_true.append(event.t)
            ys_true.append(event.y)
    
    true_traj = np.array(ys_true, dtype=float).reshape(-1, dim)
    if len(true_traj) > n_steps:
        true_traj = true_traj[:n_steps]
    
    # 4. Подготовка начального сегмента для предсказания
    # Берём первые initial_segment_length точек как "известные"
    initial_segment = true_traj[:initial_segment_length]
    time = time_train

    # выровнять длины time / true_traj / pred_traj
    n = min(len(time), len(true_traj))
    time = time[:n]
    true_traj = true_traj[:n]

    pred_traj = fdmd.predict(initial_segment, time)

    pred_traj = pred_traj[:n]

    # теперь безопасно
    L = min(initial_segment_length, n-1) 
    forecast_true = true_traj[L:]
    forecast_pred = pred_traj[L:]
    mse = np.mean((forecast_true - forecast_pred) ** 2)

    print(f"Forecast MSE (from t={time[L]:.2f}): {mse:.6e}")
    
    plt.figure(figsize=(14, 6))
    
    # Component 1
    plt.subplot(1, 2, 1)
    plt.plot(time, true_traj[:, 0], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(time[:initial_segment_length], initial_segment[:, 0], 'bo-', 
             label='Initial Segment', markersize=4, alpha=0.7)
    plt.plot(time[:len(true_traj)//2], pred_traj[:len(true_traj)//2, 0], 'r-', label='Fractional DMD Forecast', alpha=0.8)
    plt.axvline(x=time[initial_segment_length-1], color='gray', linestyle=':', 
                label='Forecast Start', alpha=0.6)
    plt.title(f"Component 1 (q={q_true})")
    plt.xlabel("Time [s]")
    plt.ylabel("State $x_1$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Component 2
    plt.subplot(1, 2, 2)
    plt.plot(time, true_traj[:, 1], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(time[:initial_segment_length], initial_segment[:, 1], 'bo-', 
             label='Initial Segment', markersize=4, alpha=0.7)
    plt.plot(time, pred_traj[:, 1], 'r-', label='Fractional DMD Forecast', alpha=0.8)
    plt.axvline(x=time[initial_segment_length-1], color='gray', linestyle=':', 
                label='Forecast Start', alpha=0.6)
    plt.title(f"Component 2 (q={q_true})")
    plt.xlabel("Time [s]")
    plt.ylabel("State $x_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Запуск
if __name__ == "__main__":
    run_test(
        q_true=0.7,
        lambda_param=-1.5,
        n_train_traj=10,
        n_steps=1000,
        T_max=1.5,
        kernel_gamma=2.0,
        regularization=1e-7,
        initial_segment_length=50  
    )
