import numpy as np
import matplotlib.pyplot as plt

from pycaputo.controller import make_fixed_controller
from pycaputo.derivatives import CaputoDerivative as D
from pycaputo.fode import caputo
from pycaputo.events import StepCompleted
from pycaputo.stepping import evolve

from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel
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
        # np.sum((x-y)**2) корректнее работает с массивами numpy любой формы
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
    Генерация набора траекторий для системы D^{q_true} y(t) = f(t, y, *f_args).
    """
    t0 = 0.0
    dt = T_max / (n_steps - 1)

    rng = np.random.default_rng(seed)
    initial_conditions = rng.uniform(ic_low, ic_high, size=(n_train_traj, dim))

    train_trajectories = []
    
    # Обертка для функции правой части
    def source_func(t, y):
        # pycaputo ожидает array
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
        # evolve возвращает генератор событий
        for event in evolve(stepper):
            if isinstance(event, StepCompleted):
                ys.append(event.y)

        # Формируем массив (n_actual_steps, dim)
        traj = np.array(ys, dtype=float).reshape(-1, dim)

        if len(traj) > n_steps:
            traj = traj[:n_steps + 1]
            
        train_trajectories.append(traj)

    # Генерируем идеальную временную сетку
    time = np.linspace(0, T_max, n_steps)
    return time, train_trajectories


# Правые части уравнений для генерации тестовых систем
def rhs_linear(t, y, lambda_param):
    return lambda_param * np.array(y)

def rhs_logistic(t, y, r):
    return r * y * (1.0 - y)

def rhs_quadratic(t, y, a, b):
    return a * y - b * y**2

def rhs_mu_cubic(t, y, mu):
    return mu * (1.0 - y**2) * y - y


# Тестовая функция для запуска всего пайплайна
def run_test(
    system_name,
    time,
    train_trajectories,
    test_traj,
    q_true,
    dim,
    kernel = RBFKernel(gamma=0.5),
    n_quad_points=100,
    regularization=1e-8,
    initial_segment_length=10,  # Длина начального сегмента для подачи в predict
    plot_part = 1.0
):
    print(f"\n=== Fractional DMD Test: {system_name} ===")
    print(f"Training trajectories: {len(train_trajectories)}")

    dt = time[1] - time[0]
    print(f"Detected dt={dt:.5f} from training data.")

    print("Fitting OKHS Transformer...")
    okhs = OKHSTransformer(
        kernel=kernel,
        q=q_true,
        n_quad_points=n_quad_points,
        dt=dt
    )
    okhs.fit(train_trajectories)


    print("Fitting Fractional Liouville Operator...")
    liouville_op = FractionalLiouvilleOperator(
        okhs_transformer=okhs, 
        n_quad_points=n_quad_points
    )
    liouville_op.fit()


    print("Fitting Fractional DMD...")
    fdmd = FractionalDMD(
        liouville_operator=liouville_op,
        n_quad_points=n_quad_points,
        regularization=regularization
    )
    fdmd.fit()

    
    initial_segment = test_traj[:initial_segment_length]
    

    pred_traj = fdmd.plot_predict(initial_segment, time)
    print(f"initial coefficients: {fdmd.initial_coefficients_}")
    print(f"Length of test_traj: {len(test_traj)}, pred_traj: {len(pred_traj)}, length of time: {len(time)}")

    # Метрики считаем только на прогнозной части (от L до конца)
    forecast_true = test_traj[initial_segment_length:]
    forecast_pred = pred_traj[initial_segment_length:]

    mse = float(np.mean((forecast_true - forecast_pred) ** 2))
    print(f"Forecast MSE (t > {time[initial_segment_length-1]:.2f}): {mse:.6e}")

    # Визуализация только части временного ряда (отбрасываем конец с большой ошибкой)
    plot_end = int(len(time) * plot_part)

    if dim == 1:
        plt.figure(figsize=(6, 4))
        plt.plot(time[:plot_end], test_traj[:plot_end, 0], "k--", label="Ground Truth", linewidth=2)
        # Отрисовка начального сегмента
        plt.plot(time[:initial_segment_length], initial_segment[:, 0], "bo-", label="Initial Segment", markersize=4, alpha=0.6)
        # Отрисовка прогноза
        plt.plot(time[:plot_end], pred_traj[:plot_end, 0], "r-", label="Forecast", alpha=0.8)
        
        plt.axvline(x=time[initial_segment_length-1], color='gray', linestyle=':', label='Start Forecast')
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
            plt.plot(time[:plot_end], test_traj[:plot_end, j], "k--", label="Ground Truth", linewidth=2)
            plt.plot(time[:initial_segment_length], initial_segment[:, j], "bo-", label="Init", markersize=4, alpha=0.6)
            plt.plot(time[:plot_end], pred_traj[:plot_end, j], "r-", label="Forecast", alpha=0.8)
            plt.axvline(x=time[initial_segment_length-1], color='gray', linestyle=':')
            
            plt.title(f"{system_name}: x_{j+1}")
            plt.xlabel("Time")
            plt.grid(True, alpha=0.3)
            if j == 0:
                plt.ylabel("State")
                plt.legend()
        plt.tight_layout()
        plt.show()

    return mse


if __name__ == "__main__":

    cases = [
        dict(
            name="Linear: D^α y = λ*y",
            f=rhs_linear,
            args=(-2.0,),
            dim=2,
            ic_low=-1.0,
            ic_high=1.0,
            x0_test=np.array([0.5, -0.5]),
            kernel_gamma=1.5,
            q_true=0.8,
            n_train_traj=20,
            n_steps_train=100,
            T_max_train=2.0,
            initial_segment_length=21,
            n_quad_points=30,
            regularization=1e-7,
            plot_part=0.5,
            seed=42
        ),
        dict(
            name="Logistic: D^α y = r*y*(1-y)",
            f=rhs_logistic,
            args=(1.5,),
            dim=1,
            ic_low=0.1,
            ic_high=0.9,
            x0_test=np.array([0.2]),
            kernel_gamma=2.0,
            q_true=0.8,
            n_train_traj=20,
            n_steps_train=100,
            T_max_train=2.0,
            initial_segment_length=21,
            n_quad_points=30,
            regularization=1e-7,
            plot_part=0.5,
            seed=42
        ),
        dict(
            name="Quadratic: D^α y = a*y - b*y²",
            f=rhs_quadratic,
            args=(2.0, 1.0),
            dim=1,
            ic_low=0.1,
            ic_high=0.9,
            x0_test=np.array([0.4]),
            kernel_gamma=3.0,
            q_true=0.8,
            n_train_traj=20,
            n_steps_train=100,
            T_max_train=2.0,
            initial_segment_length=21,
            n_quad_points=30,
            regularization=1e-7,
            plot_part=0.5,
            seed=42
        ),
        dict(
            name="Van der Pol-like",
            f=rhs_mu_cubic,
            args=(3.0,),
            dim=1,
            ic_low=-0.5,
            ic_high=0.5,
            x0_test=np.array([0.4]),
            kernel_gamma=5.0,
            q_true=0.8,
            n_train_traj=20,
            n_steps_train=100,
            T_max_train=2.0,
            initial_segment_length=21,
            n_quad_points=30,
            regularization=1e-7,
            plot_part=0.5,
            seed=42
        ),
    ]

    for cfg in cases:
        time, trajectories = generate_trajectories_pycaputo(
            cfg["f"], *cfg["args"],
            q_true=cfg["q_true"],
            n_train_traj=cfg["n_train_traj"] + 1,  # +1 для тестовой траектории
            n_steps=cfg["n_steps_train"],
            T_max=cfg["T_max_train"],
            dim=cfg["dim"],
            seed=cfg["seed"],
            ic_low=cfg["ic_low"],
            ic_high=cfg["ic_high"],
        )
        print(f"len of time: {len(time)}, shape of traj: {np.array(trajectories).shape}")
        
        # последняя траектория тестовая
        test_traj = trajectories[cfg["n_train_traj"]]
        train_trajectories = trajectories[:cfg["n_train_traj"]]

        run_test(
            system_name=cfg["name"],
            time=time,
            train_trajectories=train_trajectories,
            test_traj=test_traj,
            q_true=cfg["q_true"],
            dim=cfg["dim"],
            # kernel=OccupationKernel(q=cfg["q_true"], kernel_type='fractional'),
            kernel=RBFKernel(gamma=cfg['kernel_gamma']),
            n_quad_points=cfg["n_quad_points"],
            regularization=cfg["regularization"],
            initial_segment_length=cfg["initial_segment_length"],
            plot_part=cfg["plot_part"]
        )