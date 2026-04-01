import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Предполагается, что эти импорты у тебя уже настроены на обновленные классы
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_dmd import FractionalDMD
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.gram_transform import OKHSTransformer
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.fractional_liouville import FractionalLiouvilleOperator
from okhs_experiment_utils import ExperimentConfig, fit_okhs_fdmd_pipeline
from example_common import generate_trajectories_pycaputo

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.kernels import DeepKernel
# ==========================================
# 1. Архитектура нейросети и Deep Kernel
# ==========================================

class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=None, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype

        # Если output_dim не задан, оставляем размерность как есть
        self.out_dim = output_dim if output_dim is not None else input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.out_dim, dtype=self.dtype)
        )
        
        # Инициализируем веса и смещения последнего слоя нулями, чтобы изначально нейросеть была тождественным отображением
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x):
        residual = self.net(x)
        
        # Если размерности совпадают, просто складываем (ResNet-style)
        if x.shape[-1] == self.out_dim:
            return x + residual
            
        # Если мы проецируем в пространство бóльшей размерности:
        # дополняем исходный x нулями до нужной размерности, чтобы сохранить евклидово расстояние.
        # ||[x, 0, 0] - [y, 0, 0]|| == ||x - y||
        if self.out_dim > x.shape[-1]:
            padding = torch.zeros(
                *x.shape[:-1], self.out_dim - x.shape[-1], 
                device=x.device, dtype=x.dtype
            )
            x_padded = torch.cat([x, padding], dim=-1)
            return x_padded + residual
            
        # Защита от случая output_dim < input_dim (сжатие). 
        return x[..., :self.out_dim] + residual


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

def compare_models_and_visualize(
    deep_fdmd, 
    baseline_fdmd, 
    test_trajectories, 
    time_grid, 
    config, 
    n_plots=3
):
    """
    Вычисляет ошибки на тесте, выводит сравнительную таблицу и строит графики 
    для первых n_plots траекторий.
    """
    initial_len = config.initial_segment_length
    results = []

    print("\n" + "="*55)
    print(f"{'Траектория':<12} | {'Baseline MSE':<17} | {'Deep fDMD MSE':<17}")
    print("-" * 55)

    n_plots = min(n_plots, len(test_trajectories))
    
    # Для построения сетки графиков
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    mean_baseline_mse = 0.0
    mean_deep_mse = 0.0

    for i, test_traj in enumerate(test_trajectories):
        init_seg = test_traj[:initial_len]
        target_future = test_traj[initial_len:]
        
        # Инференс обеих моделей
        pred_deep = deep_fdmd.predict(init_seg, time_grid, return_tensor=False)
        pred_base = baseline_fdmd.predict(init_seg, time_grid) # Возвращает numpy
        
        future_deep = pred_deep[initial_len:]
        future_base = pred_base[initial_len:]
        
        # Вычисление MSE по l2-норме состояния
        mse_deep = np.mean((target_future - future_deep)**2)
        mse_base = np.mean((target_future - future_base)**2)
        
        mean_deep_mse += mse_deep
        mean_baseline_mse += mse_base

        if i < 10: # Выводим в таблицу только первые 10
            print(f"Test #{i:<7} | {mse_base:<17.4e} | {mse_deep:<17.4e}")

        # Отрисовка
        if i < n_plots:
            ax = axes[i]
            # График берет первую компоненту состояния (индекс 0), если система многомерная
            dim_idx = 0 
            
            ax.plot(time_grid, test_traj[:, dim_idx], 'k--', linewidth=2, label="Ground Truth")
            ax.plot(time_grid[:initial_len], init_seg[:, dim_idx], 'ko-', markersize=4, alpha=0.5, label="Initial Segment")
            
            ax.plot(time_grid, pred_base[:, dim_idx], 'r-', alpha=0.7, label="Baseline (RBF)")
            ax.plot(time_grid, pred_deep[:, dim_idx], 'g-', linewidth=2, alpha=0.8, label="Deep fDMD")
            
            ax.axvline(x=time_grid[initial_len-1], color='gray', linestyle=':')
            ax.set_title(f"Прогноз тестовой траектории #{i} (x_{dim_idx})")
            ax.set_xlabel("Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

    mean_baseline_mse /= len(test_trajectories)
    mean_deep_mse /= len(test_trajectories)
    
    print("-" * 55)
    print(f"{'MEAN MSE':<12} | {mean_baseline_mse:<17.4e} | {mean_deep_mse:<17.4e}")
    print("="*55 + "\n")

    plt.tight_layout()
    plt.show()


def train_deep_fdmd(
    config: ExperimentConfig, 
    trajectories: list[np.ndarray], 
    time: np.ndarray, 
    epochs: int = 50, 
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"--- Начинаем обучение Deep fDMD на {device.upper()} ---")
    
    # 1. Разделение данных
    n_total = len(trajectories)
    n_basis = int(n_total * 0.6)
    n_val = int(n_total * 0.3)
    
    basis_trajectories = trajectories[:n_basis]
    val_trajectories = trajectories[n_basis:n_basis + n_val]
    test_trajectories = trajectories[n_basis + n_val:]
    
    # 2. Инициализация нейросети и пайплайна
    feature_extractor = DeepFeatureExtractor(input_dim=config.dim, dtype=torch.float64).to(device)
    kernel = DeepKernel(feature_extractor).to(device)

    # Sanity Check
    x_sample = torch.randn(1, config.dim, dtype=torch.float64).to(device)
    y_sample = torch.randn(1, config.dim, dtype=torch.float64).to(device)
    with torch.no_grad():
        deep_k_val = kernel._compute_single_kernel(x_sample, y_sample)
        rbf_k_val = torch.sum(x_sample * y_sample).item()
        print(f"Sanity Check | Deep Kernel: {deep_k_val:.4f}, Baseline (Dot): {rbf_k_val:.4f}")

    optimizer = torch.optim.Adam(kernel.parameters(), lr=lr, weight_decay=1e-5)
    
    okhs = OKHSTransformer(
        kernel=kernel, q=config.q_true, n_quad_points=config.n_quad_points, 
        dt=float(time[1] - time[0]), device=device
    )
    liouville = FractionalLiouvilleOperator(
        okhs_transformer=okhs, n_quad_points=config.n_quad_points
    )
    fdmd = FractionalDMD(
        liouville_operator=liouville, n_quad_points=config.n_quad_points, 
        regularization=config.regularization,
        device=device
    )

    # Словарик для хранения всей истории метрик
    history = {
        "loss": [],
        "max_real_eig": [],
        "cond_g": [],
        "drift": [],
        "max_mode_norm": []
    }
    
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Шаг A: Собираем матрицы Лиувилля и Грама
        okhs.fit(basis_trajectories)
        liouville.fit()
        fdmd.fit()
        
        # Шаг B: Оценка на валидации
        epoch_loss = 0.0
        for val_traj in val_trajectories:
            initial_segment = val_traj[:config.initial_segment_length]
            target_segment = val_traj[config.initial_segment_length:]
            
            pred_tensor = fdmd.predict(
                initial_trajectory=initial_segment, 
                t_span=time, 
                return_tensor=True 
            )
            
            pred_future = pred_tensor[config.initial_segment_length:]
            target_tensor = torch.tensor(target_segment, dtype=torch.float64, device=device)
            
            loss = torch.nn.functional.mse_loss(pred_future, target_tensor)
            epoch_loss += loss
            
        epoch_loss = epoch_loss / len(val_trajectories)
        G = okhs.gram_matrix_
        
        # Создаем маску, обнуляющую диагональ
        I = torch.eye(G.shape[0], dtype=G.dtype, device=device)
        off_diagonal_mask = 1.0 - I
        
        # Считаем штраф: сумма квадратов всех элементов вне диагонали
        diversity_loss = torch.sum((G * off_diagonal_mask) ** 2) / (G.shape[0] * (G.shape[0] - 1))
        
        # Добавляем к основному лоссу с коэффициентом (например, 1e-2 или 1e-3)
        alpha_div = 1e-2
        total_loss = epoch_loss + alpha_div * diversity_loss
        
        # Шаг C: Backpropagation от суммарного лосса
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(kernel.parameters(), max_norm=1.0)
        optimizer.step()
        
        # --- БЛОК ТРЕКИНГА МЕТРИК ---
        with torch.no_grad():
            # 1. Спектр (Максимальная действительная часть лямбды)
            eig_vals = liouville.eigenvalues_
            max_real_eig = torch.max(eig_vals.real).item()
            
            # 2. Обусловленность Грама
            cond_G = okhs.gram_condition_number_
            
            # 3. Латентный дрейф (на примере первой валидационной траектории)
            # Берем норму "остатка" (residual) сети, который и есть отклонение от тождественного отображения
            val_sample = torch.tensor(val_trajectories[0], dtype=torch.float64, device=device)
            residual = kernel.feature_extractor.net(val_sample)
            latent_drift = torch.mean(torch.norm(residual, dim=-1)).item()
            
            # 4. Норма мод
            max_mode_norm = torch.max(torch.norm(fdmd.modes_, dim=-1)).item()
            
            # Сохраняем в историю
            history["loss"].append(epoch_loss.item())
            history["max_real_eig"].append(max_real_eig)
            history["cond_g"].append(cond_G)
            history["drift"].append(latent_drift)
            history["max_mode_norm"].append(max_mode_norm)

            # Обновляем прогресс-бар и выводим лог
            pbar.set_postfix({"Val MSE": f"{epoch_loss.item():.2e}", "Drift": f"{latent_drift:.4f}"})
            
            # Раскомментируй эту строку, если хочешь видеть подробный принт на каждой эпохе
            # print(f"Ep {epoch+1:02d} | MSE: {epoch_loss.item():.2e} | Max Re(λ): {max_real_eig:+.2f} | Cond(G): {cond_G:.1e} | Drift: {latent_drift:.4f} | ||ξ||: {max_mode_norm:.1e}")
        # ----------------------------

    print("Обучение завершено!")
    
    return fdmd, test_trajectories, history


def plot_training_diagnostics(history, save_path=None):
    """
    Строит визуальный дашборд со всеми собранными во время обучения метриками.
    """
    epochs = range(1, len(history["loss"]) + 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Диагностика обучения Deep fDMD", fontsize=16, fontweight='bold')
    
    # 1. MSE Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], 'b-o', markersize=4)
    ax.set_title("MSE Loss (Val)")
    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    
    # 2. Max Re(lambda)
    ax = axes[0, 1]
    ax.plot(epochs, history["max_real_eig"], 'r-o', markersize=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Граница стабильности')
    ax.set_title("Максимальная $\\Re(\\lambda)$ (Опасность взрыва)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max Re(λ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Condition Number (Матрица Грама)
    ax = axes[1, 0]
    ax.plot(epochs, history["cond_g"], 'g-o', markersize=4)
    ax.set_title("Обусловленность матрицы Грама ($Cond(G)$)")
    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cond")
    ax.grid(True, alpha=0.3)
    
    # 4. Latent Drift (Искажение геометрии)
    ax = axes[1, 1]
    ax.plot(epochs, history["drift"], 'm-o', markersize=4)
    ax.set_title("Латентный дрейф (Отклонение от $f(x)=x$)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Residual Norm")
    ax.grid(True, alpha=0.3)
    
    # 5. Max Mode Norm ||xi||
    ax = axes[2, 0]
    ax.plot(epochs, history["max_mode_norm"], 'c-o', markersize=4)
    ax.set_title("Максимальная норма моды ($||\\xi||$)")
    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)
    
    # 6. Пустой график или текстовая сводка
    ax = axes[2, 1]
    ax.axis('off')
    summary_text = (
        f"--- ИТОГИ ОБУЧЕНИЯ ---\n\n"
        f"Start MSE: {history['loss'][0]:.4e}\n"
        f"End MSE:   {history['loss'][-1]:.4e}\n\n"
        f"Max Cond(G): {max(history['cond_g']):.2e}\n"
        f"Final Drift: {history['drift'][-1]:.4f}\n"
        f"Stable at End?: {'YES' if history['max_real_eig'][-1] <= 0 else 'NO'}"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def rhs_logistic(t, y, r):
    return r * y * (1.0 - y)

if __name__ == "__main__":
    config = ExperimentConfig(
        name="Logistic: D^α y = r*y*(1-y)",
        q_true=0.8,
        dim=1,
        n_train_traj=150, # Увеличим для нормального разделения (Basis/Val/Test)
        n_steps_train=300,
        T_max_train=3.0,
        initial_segment_length=100,
        n_quad_points=20,
        regularization=1e-3,
        plot_part=1.0,
        seed=42,
        ic_low=0.1,
        ic_high=0.9,
    )
    
    # 1. Генерируем данные
    time, trajectories = generate_trajectories_pycaputo(
        rhs_logistic,
        (1.5,),
        q_true=config.q_true,
        n_trajectories=config.n_train_traj,
        n_steps=config.n_steps_train,
        T_max=config.T_max_train,
        dim=config.dim,
        seed=config.seed,
        ic_low=config.ic_low,
        ic_high=config.ic_high,
    )
    
    # 2. Запускаем обучение
    trained_deep_fdmd, test_traj, history = train_deep_fdmd(
        config=config, 
        trajectories=trajectories, 
        time=time, 
        epochs=10, 
        lr=1e-5,
    )

    # 3. Построение графика Loss
    plot_training_loss(history["loss"])
    
    deep_kernel_module = trained_deep_fdmd.okhs.kernel
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "deep_kernel_weights.pth")
    torch.save(deep_kernel_module.state_dict(), model_path)
    print(f"Веса модели сохранены в: {model_path}")


    print("\nОбучение Baseline (обычный RBF Kernel)...")
    n_basis = int(len(trajectories) * 0.6)
    basis_trajectories = trajectories[:n_basis]
    
    # Используем твою существующую функцию для сборки стандартного пайплайна
    from example_common import RBFKernel

    baseline_artifacts = fit_okhs_fdmd_pipeline(
        time=time,
        train_trajectories=basis_trajectories,
        q_true=config.q_true,
        kernel=RBFKernel(gamma=1.0), # Фиксированное ядро без нейросети
        n_quad_points=config.n_quad_points,
        regularization=config.regularization,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    )
    baseline_fdmd = baseline_artifacts.fdmd
    
    # 5. Сравнение моделей и графики
    compare_models_and_visualize(
        deep_fdmd=trained_deep_fdmd,
        baseline_fdmd=baseline_fdmd,
        test_trajectories=test_traj,
        time_grid=time,
        config=config,
        n_plots=3 # Сколько траекторий отрисовать
    )

    # 3. Вывод дашборда
    plot_training_diagnostics(history)
