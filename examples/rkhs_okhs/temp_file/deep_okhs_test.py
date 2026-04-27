import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_dmd import FractionalDMD, plot_forecast_diagnostics
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.gram_transform import OKHSTransformer
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_liouville import FractionalLiouvilleOperator
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import CURDecomposition
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.kernels import DeepKernel
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fractional_loss import DeepFractionalDMDLoss
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.deep_fdmd_net import DeepFDMDAutoencoder

from okhs_experiment_utils import ExperimentConfig
from example_common import generate_trajectories_pycaputo, RBFKernel


class EncoderAdapter(nn.Module):
    """Обертка для передачи метода encode() внутрь DeepKernel как forward()"""

    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        return self.autoencoder.encode_trajectory(x)


def plot_training_loss(loss_history, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', markersize=4)
    plt.title("Deep fDMD: Зависимость MSE от эпохи (Val Data)")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compare_models_and_visualize(deep_fdmd, baseline_fdmd, test_trajectories, time_grid, config, n_plots=3):
    """
    Вычисляет ошибки на тесте, выводит сравнительную таблицу и строит графики
    для первых n_plots траекторий.
    """
    initial_len = config.initial_segment_length

    print("\n" + "=" * 55)
    print(f"{'Траектория':<12} | {'Baseline MSE':<17} | {'Deep fDMD MSE':<17}")
    print("-" * 55)

    n_plots = min(n_plots, len(test_trajectories))
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    mean_baseline_mse = 0.0
    mean_deep_mse = 0.0

    for i, test_traj in enumerate(test_trajectories):
        init_seg = test_traj[:initial_len]
        target_future = test_traj[initial_len:]

        pred_deep = deep_fdmd.predict(init_seg, time_grid, return_tensor=False)
        pred_base = baseline_fdmd.predict(init_seg, time_grid, return_tensor=False)

        future_deep = pred_deep[initial_len:]
        future_base = pred_base[initial_len:]
        mse_deep = np.mean((target_future - future_deep)**2)
        mse_base = np.mean((target_future - future_base)**2)

        mean_deep_mse += mse_deep
        mean_baseline_mse += mse_base

        if i < 10:
            print(f"Test #{i:<7} | {mse_base:<17.4e} | {mse_deep:<17.4e}")

        if i < n_plots:
            ax = axes[i]
            dim_idx = 0
            ax.plot(time_grid, test_traj[:, dim_idx], 'k--', linewidth=2, label="Ground Truth")
            ax.plot(time_grid[:initial_len], init_seg[:, dim_idx], 'ko-',
                    markersize=4, alpha=0.5, label="Initial Segment")
            ax.plot(time_grid, pred_base[:, dim_idx], 'r-', alpha=0.7, label="Baseline (RBF)")
            ax.plot(time_grid, pred_deep[:, dim_idx], 'g-', linewidth=2, alpha=0.8, label="Deep fDMD")
            ax.axvline(x=time_grid[initial_len - 1], color='gray', linestyle=':')
            ax.set_title(f"Прогноз тестовой траектории #{i} (x_{dim_idx})")
            ax.set_xlabel("Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

    mean_baseline_mse /= len(test_trajectories)
    mean_deep_mse /= len(test_trajectories)

    print("-" * 55)
    print(f"{'MEAN MSE':<12} | {mean_baseline_mse:<17.4e} | {mean_deep_mse:<17.4e}")
    print("=" * 55 + "\n")

    plt.tight_layout()
    plt.show()


def evaluate_full_pipeline(autoencoder, basis_trajectories, val_trajectories, time_grid, config, device):
    """Изолированный прогон пайплайна для оценки реального качества (выполняется в no_grad)."""
    autoencoder.eval()

    kernel = DeepKernel(feature_extractor=EncoderAdapter(autoencoder), base_kernel=None).to(device)
    okhs = OKHSTransformer(
        kernel=kernel,
        q=config.q_true,
        n_quad_points=config.n_quad_points,
        dt=float(
            time_grid[1] -
            time_grid[0]),
        device=device)
    liouville = FractionalLiouvilleOperator(okhs_transformer=okhs, n_quad_points=config.n_quad_points)
    fdmd = FractionalDMD(
        liouville_operator=liouville,
        n_quad_points=config.n_quad_points,
        regularization=config.regularization,
        device=device)

    with torch.no_grad():
        okhs.fit(basis_trajectories)
        liouville.fit()
        fdmd.fit()

        val_mse = 0.0
        for val_traj in val_trajectories:
            init_seg = val_traj[:config.initial_segment_length]
            target_seg = torch.tensor(val_traj[config.initial_segment_length:], dtype=torch.float64, device=device)

            pred_tensor = fdmd.predict(initial_trajectory=init_seg, t_span=time_grid, return_tensor=True)
            pred_future = pred_tensor[config.initial_segment_length:]
            val_mse += torch.nn.functional.mse_loss(pred_future, target_seg).item()
        val_mse /= len(val_trajectories)

        # Метрики стабильности и диагностики
        eig_vals = liouville.eigenvalues_
        max_real_eig = torch.max(eig_vals.real).item()
        cond_G = okhs.gram_condition_number_
        max_mode_norm = torch.max(torch.norm(fdmd.modes_, dim=-1)).item()

        # Латентный дрейф (отклонение от тождественного)
        val_sample = torch.tensor(val_trajectories[0], dtype=torch.float64, device=device)
        z_val = autoencoder.encode_trajectory(val_sample)
        latent_drift = torch.mean(torch.norm(z_val[..., :config.dim] - val_sample, dim=-1)).item()

    return val_mse, max_real_eig, cond_G, latent_drift, max_mode_norm


def train_deep_fdmd_pipeline(config, trajectories, time_grid, epochs=100, lr=1e-3, lambda_adj=1.0, device='cuda'):
    print(f"\n--- Инициализация Deep fDMD на {device.upper()} ---")

    n_basis = int(len(trajectories) * 0.6)
    n_val = int(len(trajectories) * 0.3)

    raw_basis = trajectories[:n_basis]
    val_traj = trajectories[n_basis:n_basis + n_val]
    test_traj = trajectories[n_basis + n_val:]

    cur_decomposer = CURDecomposition(params={'tolerance': [0.1], "rank": 10})
    cur_decomposer.fit_transform(np.array([traj.flatten() for traj in raw_basis]))
    basis_traj = [raw_basis[i] for i in np.sort(cur_decomposer.row_indices)]
    print(f"Выбрано {len(basis_traj)} базисных траекторий из {len(raw_basis)} (CUR Decomposition)")

    x_train_tensor = torch.tensor(np.array(basis_traj), dtype=torch.float64, device=device)
    t_grid_tensor = torch.tensor(time_grid, dtype=torch.float64, device=device)
    T_tensor = torch.full((x_train_tensor.shape[0],), time_grid[-1], dtype=torch.float64, device=device)

    latent_dim = max(config.dim, 16)
    model = DeepFDMDAutoencoder(input_dim=config.dim, latent_dim=latent_dim, dtype=torch.float64).to(device)
    adjoint_loss_fn = DeepFractionalDMDLoss(
        latent_dim=latent_dim,
        q=config.q_true,
        n_quad_points=config.n_quad_points,
        device=device)
    recon_loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(list(model.parameters()) +
                                 list(adjoint_loss_fn.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    history = {
        "loss": [], "max_real_eig": [], "cond_g": [], "drift": [], "max_mode_norm": [],
        "train_recon": [], "train_adj": []
    }

    best_val_mse = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    pbar = tqdm(range(1, epochs + 1), desc="Training Phase")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Быстрый шаг градиентного спуска (только автоэнкодер + Adjoint Loss)
        z_traj, x_recon = model(x_train_tensor)
        loss_recon = recon_loss_fn(x_recon, x_train_tensor)
        loss_adj = adjoint_loss_fn(t_grid_tensor, z_traj, T_tensor)

        loss = loss_recon + lambda_adj * loss_adj
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history['train_recon'].append(loss_recon.item())
        history['train_adj'].append(loss_adj.item())

        # Валидация полного пайплайна для дашборда
        val_mse, max_real, cond_g, drift, max_norm = evaluate_full_pipeline(
            model, basis_traj, val_traj, time_grid, config, device
        )

        history["loss"].append(val_mse)
        history["max_real_eig"].append(max_real)
        history["cond_g"].append(cond_g)
        history["drift"].append(drift)
        history["max_mode_norm"].append(max_norm)

        scheduler.step(val_mse)
        pbar.set_postfix({'Val MSE': f"{val_mse:.2e}", 'Cond(G)': f"{cond_g:.1e}"})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"\nОбучение завершено. Лучший Val MSE: {best_val_mse:.4e}")
    model.load_state_dict(best_model_state)

    return model, basis_traj, test_traj, history


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


if __name__ == "__main__":
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    cases = [
        # dict(
        #     config=ExperimentConfig(
        #         name="Lotka-Volterra (q=0.9)",
        #         q_true=0.9,
        #         dim=2,
        #         n_train_traj=100,
        #         n_steps_train=500,
        #         T_max_train=12.0,
        #         initial_segment_length=100,
        #         n_quad_points=20,
        #         regularization=1e-4,
        #         plot_part=1.0,
        #         seed=1337,
        #         ic_low=0.5,             # Популяции строго положительны
        #         ic_high=3.0,
        #     ),
        #     dynamics_func=rhs_lotka_volterra,
        #     dynamics_args=(1.5, 1.0, 1.0, 3.0),  # alpha, beta, delta, gamma
        #     kernel=RBFKernel(gamma=1.0),
        # ),
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
        config = case["config"]
        print(f"\n\n=== Запуск эксперимента: {config.name} ===")
        time_grid, trajectories = generate_trajectories_pycaputo(
            case["dynamics_func"],
            *case["dynamics_args"],
            n_trajectories=config.n_train_traj,
            q_true=config.q_true,
            n_steps=config.n_steps_train,
            T_max=config.T_max_train,
            dim=config.dim,
            seed=config.seed,
            ic_low=config.ic_low,
            ic_high=config.ic_high,
        )

        # 1. Обучение нейросети
        trained_autoencoder, basis_trajectories, test_traj, history = train_deep_fdmd_pipeline(
            config=config,
            trajectories=trajectories,
            time_grid=time_grid,
            epochs=25,
            lr=1e-3,
            device=device_str
        )

        # 2. Вывод дашбордов обучения
        plot_training_loss(history["loss"])
        # plot_training_diagnostics(history)

        # 3. Сохранение весов энкодера
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "deep_autoencoder_weights.pth")
        torch.save(trained_autoencoder.state_dict(), model_path)
        print(f"Веса модели сохранены в: {model_path}")

        # 4. Сборка финального пайплайна Deep fDMD для тестирования
        print("\nСборка финального пайплайна Deep fDMD...")
        trained_autoencoder.eval()
        deep_kernel = DeepKernel(feature_extractor=EncoderAdapter(trained_autoencoder), base_kernel=None).to(device)
        okhs_deep = OKHSTransformer(
            kernel=deep_kernel,
            q=config.q_true,
            n_quad_points=config.n_quad_points,
            dt=float(
                time_grid[1] -
                time_grid[0]),
            device=device)
        okhs_deep.fit(basis_trajectories)  # Обучаем базовое OKHS для сравнения Gram матриц
        liouville_deep = FractionalLiouvilleOperator(okhs_transformer=okhs_deep, n_quad_points=config.n_quad_points)
        liouville_deep.fit()
        trained_deep_fdmd = FractionalDMD(
            liouville_operator=liouville_deep,
            n_quad_points=config.n_quad_points,
            regularization=config.regularization,
            device=device)
        trained_deep_fdmd.fit(basis_trajectories)

        # 5. Обучение Baseline (RBF Kernel)
        print("Обучение Baseline (обычный RBF Kernel)...")
        rbf_kernel = RBFKernel(gamma=1.0)
        okhs_base = OKHSTransformer(
            kernel=rbf_kernel,
            q=config.q_true,
            n_quad_points=config.n_quad_points,
            dt=float(
                time_grid[1] -
                time_grid[0]),
            device=device)
        okhs_base.fit(basis_trajectories)
        liouville_base = FractionalLiouvilleOperator(okhs_transformer=okhs_base, n_quad_points=config.n_quad_points)
        liouville_base.fit()
        baseline_fdmd = FractionalDMD(
            liouville_operator=liouville_base,
            n_quad_points=config.n_quad_points,
            regularization=config.regularization,
            device=device)
        baseline_fdmd.fit(basis_trajectories)

        # 6. Сравнение моделей и финальные графики на тестовой выборке
        compare_models_and_visualize(
            deep_fdmd=trained_deep_fdmd,
            baseline_fdmd=baseline_fdmd,
            test_trajectories=test_traj,
            time_grid=time_grid,
            config=config,
            n_plots=3
        )

        # print("Gram matrix (Deep OKHS):")
        # print(okhs_deep.gram_matrix_)
        # print(f"Condition Number of Gram Matrix: {okhs_deep.gram_condition_number_:.2e}")

        # Опционально: Диагностика устойчивости мод (спектр)
        sample_idx = 0
        init_seg = test_traj[sample_idx][:config.initial_segment_length]
        plot_forecast_diagnostics(
            fdmd=trained_deep_fdmd,
            initial_trajectory=init_seg,
            t_span=time_grid,
            stability_threshold=0
        )
