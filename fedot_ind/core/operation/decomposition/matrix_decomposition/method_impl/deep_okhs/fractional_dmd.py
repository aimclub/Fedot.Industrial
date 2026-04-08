
import warnings
import numpy as np
from scipy.special import roots_jacobi
from sklearn.base import BaseEstimator, RegressorMixin
import torch

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.mittag_leffler import _mittag_leffler_tensor

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.policies import RegularizationPolicy, StabilityPolicy
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.matrix_utils import (
     validate_initial_coefficient_feasibility,
)
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.matrix_utils import select_stable_modes



class FractionalDMD(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            liouville_operator,
            n_quad_points=20,
            regularization=1e-8,
            regularization_policy=None,
            stability_policy=None,
            verbose=False,
            device = 'cpu',
    ):
        self.liouville_operator = liouville_operator
        self.okhs = liouville_operator.okhs
        self.n_quad_points = n_quad_points
        self.regularization = regularization
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.stability_policy = stability_policy or StabilityPolicy()
        self.verbose = verbose
        
        if device == 'cuda' and not torch.cuda.is_available():
            if self.verbose:
                print("Warning: CUDA is not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.modes_ = None
        self._quad_cache = None

    def _normalize_state_trajectory(self, trajectory):
        if not isinstance(trajectory, torch.Tensor):
            normalized = torch.tensor(trajectory, dtype=torch.float64, device=self.device)
        else:
            normalized = trajectory.to(torch.float64).to(self.device)
            
        if normalized.ndim == 1:
            return normalized.view(-1, 1)
        return normalized

    @staticmethod
    def _build_mode_groups(eigenvalues, tolerance=1e-8):
        groups = []
        used = set()
        eig_np = eigenvalues.detach().cpu().numpy()
        for index, eigenvalue in enumerate(eig_np):
            if index in used:
                continue
            if abs(np.imag(eigenvalue)) <= tolerance:
                groups.append((index,))
                used.add(index)
                continue
            conjugate_index = None
            for candidate_index in range(index + 1, len(eig_np)):
                if candidate_index in used:
                    continue
                candidate_value = eig_np[candidate_index]
                if np.isclose(candidate_value, np.conj(eigenvalue), atol=tolerance, rtol=1e-6):
                    conjugate_index = candidate_index
                    break
            if conjugate_index is None:
                groups.append((index,))
                used.add(index)
            else:
                groups.append((index, conjugate_index))
                used.add(index)
                used.add(conjugate_index)
        return groups

    @staticmethod
    def _resolve_prediction_mode_budget(available_equations, horizon, max_prediction_modes=None, min_prediction_modes=4):
        if max_prediction_modes is not None:
            requested_budget = int(max_prediction_modes)
        else:
            requested_budget = max(
                int(min_prediction_modes),
                int(np.ceil(np.sqrt(max(available_equations, 1)))),
                int(np.ceil(max(horizon, 1) / 2.0)),
            )
            requested_budget = min(requested_budget, 12)
        return max(1, min(int(requested_budget), int(available_equations)))

    def _select_prediction_modes(
            self,
            initial_trajectory,
            t_span,
            eig,
            xi,
            stable_indices,
            available_equations,
            prediction_mode_selection_policy="all_stable",
            max_prediction_modes=None,
            min_prediction_modes=4,
    ):
        policy = str(prediction_mode_selection_policy or "all_stable").lower()
        horizon = len(t_span)
        all_indices = torch.arange(len(eig), device=eig.device)
        default_budget = min(len(eig), max(1, available_equations))

        if policy in {"all_stable", "none"}:
            selected = all_indices[:default_budget]
            return {
                "eig": eig[selected],
                "xi": xi[selected],
                "selected_local_indices": selected,
                "selected_global_indices": stable_indices[selected],
                "policy": policy,
                "prediction_mode_budget": default_budget,
                "preselection_count": len(selected),
                "group_count": len(selected),
                "score_table": [],
            }

        budget = self._resolve_prediction_mode_budget(
            available_equations=available_equations,
            horizon=horizon,
            max_prediction_modes=max_prediction_modes,
            min_prediction_modes=min_prediction_modes,
        )
        if len(eig) <= budget:
            selected = all_indices
            return {
                "eig": eig[selected],
                "xi": xi[selected],
                "selected_local_indices": selected,
                "selected_global_indices": stable_indices[selected],
                "policy": policy,
                "prediction_mode_budget": budget,
                "preselection_count": len(selected),
                "group_count": len(selected),
                "score_table": [],
            }

        groups = self._build_mode_groups(eig)
        group_descriptors = []
        for group in groups:
            group_tensor = torch.tensor(group, dtype=torch.long, device=eig.device)
            group_eig = eig[group_tensor]
            group_xi = xi[group_tensor]
            
            # Предварительный скоринг с использованием торч
            norm_sum = torch.sum(torch.linalg.norm(group_xi, dim=1))
            imag_sum = torch.sum(torch.abs(group_eig.imag))
            real_sum = torch.sum(torch.clamp(group_eig.real, min=0.0))
            
            preliminary_score = norm_sum * (1.0 + 0.5 * imag_sum + 0.25 * real_sum)
            
            descriptor = {
                "indices": tuple(int(item) for item in group),
                "size": int(len(group)),
                "preliminary_score": preliminary_score.item(),
            }
            group_descriptors.append(descriptor)

        preselection_budget = min(len(eig), max(budget, min(int(available_equations), int(max(budget * 3, budget)))))
        preliminary_sorted = sorted(group_descriptors, key=lambda item: item["preliminary_score"], reverse=True)
        
        preselected_local_indices = []
        preselected_groups = []
        for descriptor in preliminary_sorted:
            if len(preselected_local_indices) + descriptor["size"] > preselection_budget and preselected_groups:
                continue
            preselected_groups.append(descriptor)
            preselected_local_indices.extend(descriptor["indices"])
            if len(preselected_local_indices) >= preselection_budget:
                break

        preselected_local_indices = torch.tensor(sorted(set(preselected_local_indices)), dtype=torch.long, device=eig.device)
        preselected_eig = eig[preselected_local_indices]
        preselected_xi = xi[preselected_local_indices]

        coefficients = self.fit_initial_coefficients(initial_trajectory, eig=preselected_eig, Xi=preselected_xi)
        q_value = self.okhs.q
        
        tail_points = min(max(4, horizon // 2), len(initial_trajectory))
        tail_start = len(initial_trajectory) - tail_points
        
        device = eig.device
        tail_grid = torch.arange(tail_start, len(initial_trajectory), dtype=torch.float64, device=device) * self.okhs.dt
        future_grid = torch.tensor(t_span, dtype=torch.float64, device=device)
        
        tail_ml = _mittag_leffler_tensor(preselected_eig.unsqueeze(0) * (tail_grid.to(torch.complex128) ** q_value).unsqueeze(1), q_value)
        future_ml = _mittag_leffler_tensor(preselected_eig.unsqueeze(0) * (future_grid.to(torch.complex128) ** q_value).unsqueeze(1), q_value)

        local_index_map = {int(global_idx): pos for pos, global_idx in enumerate(preselected_local_indices.tolist())}
        scored_groups = []
        
        for descriptor in preselected_groups:
            local_positions = torch.tensor([local_index_map[idx] for idx in descriptor["indices"]], dtype=torch.long, device=device)
            
            tail_comp = (tail_ml[:, local_positions].unsqueeze(2) * coefficients[local_positions].unsqueeze(0).unsqueeze(2) * preselected_xi[local_positions].unsqueeze(0)).real
                         
            future_comp = (future_ml[:, local_positions].unsqueeze(2) * coefficients[local_positions].unsqueeze(0).unsqueeze(2) * preselected_xi[local_positions].unsqueeze(0)).real

            tail_energy = torch.mean(torch.abs(tail_comp)).item()
            future_energy = torch.mean(torch.abs(future_comp)).item()
            future_variability = torch.mean(torch.std(future_comp, dim=0)).item()
            oscillation_score = torch.sum(torch.abs(preselected_eig[local_positions].imag)).item()
            
            refined_score = 0.6 * tail_energy + 0.4 * future_variability + 0.2 * future_energy + 0.1 * oscillation_score
            scored_groups.append({
                "indices": descriptor["indices"],
                "size": descriptor["size"],
                "tail_energy": tail_energy,
                "future_energy": future_energy,
                "future_variability": future_variability,
                "oscillation_score": oscillation_score,
                "refined_score": refined_score,
            })

        scored_groups.sort(key=lambda item: item["refined_score"], reverse=True)
        selected_local_indices_list = []
        for descriptor in scored_groups:
            if len(selected_local_indices_list) + descriptor["size"] > budget and selected_local_indices_list:
                continue
            selected_local_indices_list.extend(descriptor["indices"])
            if len(selected_local_indices_list) >= budget:
                break

        if len(selected_local_indices_list) < min(int(min_prediction_modes), len(preselected_local_indices)):
            for descriptor in scored_groups:
                for idx in descriptor["indices"]:
                    if idx not in selected_local_indices_list:
                        selected_local_indices_list.append(idx)
                if len(selected_local_indices_list) >= min(int(min_prediction_modes), len(preselected_local_indices)):
                    break

        selected_local_indices = torch.tensor(sorted(set(selected_local_indices_list)), dtype=torch.long, device=eig.device)
        return {
            "eig": eig[selected_local_indices],
            "xi": xi[selected_local_indices],
            "selected_local_indices": selected_local_indices,
            "selected_global_indices": stable_indices[selected_local_indices],
            "policy": policy,
            "prediction_mode_budget": budget,
            "preselection_count": int(len(preselected_local_indices)),
            "group_count": int(len(groups)),
            "score_table": [
                {
                    "indices": list(item["indices"]),
                    "refined_score": float(item["refined_score"]),
                    "tail_energy": float(item["tail_energy"]),
                    "future_energy": float(item["future_energy"]),
                    "future_variability": float(item["future_variability"]),
                    "oscillation_score": float(item["oscillation_score"]),
                }
                for item in scored_groups[:10]
            ],
        }

    def _prepare_prediction_state(
            self,
            initial_trajectory,
            t_span,
            stability_threshold=None,
            prediction_mode_selection_policy="all_stable",
            max_prediction_modes=None,
            min_prediction_modes=4,
    ):
        initial_trajectory = self._normalize_state_trajectory(initial_trajectory)
        t_span = torch.tensor(t_span, dtype=torch.float64, device=initial_trajectory.device).view(-1)

        eig_full = self.liouville_operator.eigenvalues_
        xi_full = self.modes_

        stable_mask = select_stable_modes(
            eig_full,
            stability_policy=self.stability_policy,
            stability_threshold=stability_threshold,
        )
        eig = eig_full[stable_mask]
        xi = xi_full[stable_mask]
        if np.asarray(eig).size == 0:
            raise ValueError("No stable modes remain after applying the stability policy.")

        available_equations = int(initial_trajectory.shape[0] * initial_trajectory.shape[1])
        selected_mode_indices = torch.where(stable_mask)[0]
        
        selection = self._select_prediction_modes(
            initial_trajectory=initial_trajectory,
            t_span=t_span.cpu().numpy(), # Для horizon length
            eig=eig,
            xi=xi,
            stable_indices=selected_mode_indices,
            available_equations=available_equations,
            prediction_mode_selection_policy=prediction_mode_selection_policy,
            max_prediction_modes=max_prediction_modes,
            min_prediction_modes=min_prediction_modes,
        )
        
        eig = selection["eig"]
        xi = selection["xi"]
        selected_mode_indices = selection["selected_global_indices"]
        prediction_mode_cap = int(selection["prediction_mode_budget"])
        prediction_mode_cap_applied = bool(len(torch.where(stable_mask)[0]) > len(selected_mode_indices))

        coefficients = self.fit_initial_coefficients(initial_trajectory, Xi=xi, eig=eig)
        
        t_q = (t_span.to(torch.complex128) ** self.okhs.q).unsqueeze(1)
        lam = eig.unsqueeze(0)
        mittag = _mittag_leffler_tensor(lam * t_q, self.okhs.q)
        
        predicted = (mittag @ (coefficients.unsqueeze(1) * xi)).real
        
        return {
            "initial_trajectory": initial_trajectory,
            "t_span": t_span,
            "eig_full": eig_full,
            "stable_mask": stable_mask,
            "eig": eig,
            "xi": xi,
            "selected_mode_indices": selected_mode_indices,
            "available_equations": available_equations,
            "prediction_mode_cap": prediction_mode_cap,
            "prediction_mode_cap_applied": prediction_mode_cap_applied,
            "prediction_mode_selection_policy": selection["policy"],
            "prediction_preselection_count": selection["preselection_count"],
            "prediction_group_count": selection["group_count"],
            "prediction_mode_scores": selection["score_table"],
            "coefficients": coefficients,
            "predicted": predicted,
        }

    def _build_prediction_diagnostics(self, state):
        initial_trajectory = state["initial_trajectory"]
        eig = state["eig"]
        xi = state["xi"]
        coefficients = state["coefficients"]
        predicted = state["predicted"]
        stable_mask = state["stable_mask"]
        selected_mode_indices = state["selected_mode_indices"]
        device = initial_trajectory.device

        initial_grid = torch.arange(initial_trajectory.shape[0], dtype=torch.float64, device=device) * self.okhs.dt
        initial_t_q = (initial_grid.to(torch.complex128) ** self.okhs.q).unsqueeze(1)
        initial_mittag = _mittag_leffler_tensor(eig.unsqueeze(0) * initial_t_q, self.okhs.q)
        reconstruction = (initial_mittag @ (coefficients.unsqueeze(1) * xi)).real

        last_observed = initial_trajectory[-1].view(-1)
        first_prediction = predicted[0].view(-1) if len(predicted) else last_observed
        discontinuity = first_prediction - last_observed

        return {
            "n_total_modes": int(len(state["eig_full"])),
            "n_stable_modes": int(torch.sum(stable_mask).item()),
            "stable_mode_indices": torch.where(stable_mask)[0].tolist(),
            "n_selected_prediction_modes": int(len(eig)),
            "selected_prediction_mode_indices": selected_mode_indices.tolist(),
            "available_equations": int(state["available_equations"]),
            "prediction_mode_cap": int(state["prediction_mode_cap"]),
            "prediction_mode_cap_applied": bool(state["prediction_mode_cap_applied"]),
            "prediction_mode_selection_policy": state["prediction_mode_selection_policy"],
            "prediction_preselection_count": int(state["prediction_preselection_count"]),
            "prediction_group_count": int(state["prediction_group_count"]),
            "prediction_mode_scores": state["prediction_mode_scores"],
            "stable_eigenvalues_real": eig.real.tolist(),
            "stable_eigenvalues_imag": eig.imag.tolist(),
            "mode_norms": torch.linalg.norm(xi, dim=1).tolist() if len(xi) else [],
            "initial_coefficients_real": coefficients.real.tolist(),
            "initial_coefficients_imag": coefficients.imag.tolist(),
            "initial_reconstruction_rmse": float(torch.sqrt(torch.mean((reconstruction - initial_trajectory) ** 2)).item()),
            "initial_reconstruction_mae": float(torch.mean(torch.abs(reconstruction - initial_trajectory)).item()),
            "last_observed_value": last_observed.tolist(),
            "first_prediction_value": first_prediction.tolist(),
            "boundary_discontinuity": discontinuity.tolist(),
            "boundary_discontinuity_abs_mean": float(torch.mean(torch.abs(discontinuity)).item()),
        }

    def _get_jacobi_rule(self):
        if self._quad_cache is None:
            nodes, weights = roots_jacobi(self.n_quad_points, self.okhs.q - 1, 0)
            self._quad_cache = (
                torch.tensor(nodes, dtype=torch.float64),
                torch.tensor(weights, dtype=torch.float64)
            )
        return self._quad_cache
        
    def compute_identity_projections(self, trajectories):
        """
        Векторизованное вычисление матрицы Y (проекции g_id на occupation kernels).
        """
        n_traj = len(trajectories)
        n_features = trajectories[0].shape[1]
        device = self.liouville_operator.eigenvectors_.device
        
        Y = torch.zeros((n_traj, n_features), dtype=torch.float64, device=device)
        _, weights = self._get_jacobi_rule()
        weights = weights.to(device)
        
        for k in range(n_traj):
            traj = self._normalize_state_trajectory(trajectories[k]).to(device)
            T_traj = self.okhs._get_trajectory_duration(traj)
            
            if T_traj <= 1e-14:
                continue
                
            # Запрашиваем значения разом во всех узлах
            vals = self.okhs._evaluate_trajectory_at_nodes(traj, T_traj) # (n_quad, n_features)
            jacobian_factor = (T_traj / 2.0) ** self.okhs.q
            
            # Эквивалент взвешенной суммы интеграла
            integral_sum = torch.einsum('k,kd->d', weights, vals)
            Y[k, :] = self.okhs.C_q * jacobian_factor * integral_sum
            
        return Y

    def compute_eigenfunction_projections(self, Y, V):
        """ B = V^* Y """
        return V.conj().T @ Y

    def solve_modes(self, W, B):
        """ Решение системы W * Xi = B. """
        regularization = max(self.regularization, self.regularization_policy.base_jitter)
        W_reg = W + regularization * torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
        
        try:
            Xi = torch.linalg.solve(W_reg, B)
        except RuntimeError:
            Xi = torch.linalg.pinv(W_reg) @ B
        return Xi

    def fit(self, trajectories=None):
        if trajectories is None:
            trajectories = self.okhs.train_trajectories_
            
        if self.liouville_operator.eigenvectors_ is None:
             raise ValueError("Liouville operator must be fitted.")
             
        V = self.liouville_operator.eigenvectors_
        G = self.okhs.gram_matrix_.to(V.dtype)
        
        Y = self.compute_identity_projections(trajectories).to(V.dtype)
        B = self.compute_eigenfunction_projections(Y, V)
        W = V.conj().T @ G @ V
        
        self.modes_ = self.solve_modes(W, B)
        return self

    def fit_initial_coefficients(self, initial_trajectory, eig=None, Xi=None):
        """
        Векторизованная сборка матрицы наименьших квадратов и решение.
        """
        initial_trajectory = self._normalize_state_trajectory(initial_trajectory)
        K, n_features = initial_trajectory.shape
        device = initial_trajectory.device

        if eig is None:
            eig = self.liouville_operator.eigenvalues_
        if Xi is None:
            Xi = self.modes_
            
        n_modes = len(eig)
        validate_initial_coefficient_feasibility(initial_trajectory, n_modes)
        
        t_grid = torch.arange(K, dtype=torch.float64, device=device) * self.okhs.dt
        
        ml_evals = _mittag_leffler_tensor(eig.unsqueeze(0) * (t_grid.to(torch.complex128).unsqueeze(1) ** self.okhs.q), self.okhs.q)
        
        # Быстрая сборка матрицы A через тензорное перемножение: (K, n_modes, n_features)
        A_blocks = ml_evals.unsqueeze(2) * Xi.unsqueeze(0)
        
        # Транспонируем в (K, n_features, n_modes) и вытягиваем
        A = A_blocks.transpose(1, 2).reshape(K * n_features, n_modes)
        b = initial_trajectory.view(-1).to(torch.complex128)

        alpha = max(self.regularization, self.regularization_policy.base_jitter)
        
        A_stack = A.conj().T @ A
        reg_matrix = A_stack + alpha * torch.eye(n_modes, dtype=A_stack.dtype, device=device)
        b_stack = A.conj().T @ b
        
        try:
            c = torch.linalg.solve(reg_matrix, b_stack)
        except RuntimeError:
            c = torch.linalg.pinv(reg_matrix) @ b_stack

        self.initial_coefficients_ = c
        return c

    def predict(self, initial_trajectory, t_span, stability_threshold=None, return_tensor = False):
        initial_trajectory = self._normalize_state_trajectory(initial_trajectory)
        device = initial_trajectory.device
        t_span = torch.tensor(t_span, dtype=torch.float64, device=device).view(-1)
        
        eig_full = self.liouville_operator.eigenvalues_
        Xi_full = self.modes_

        stable_mask = select_stable_modes(
            eig_full,
            stability_policy=self.stability_policy,
            stability_threshold=stability_threshold,
        )
        eig = eig_full[stable_mask]
        Xi = Xi_full[stable_mask]
        if np.asarray(eig.cpu().detach().numpy()).size == 0:
            raise ValueError("No stable modes remain after applying the stability policy.")

        c = self.fit_initial_coefficients(initial_trajectory, Xi=Xi, eig=eig)
        
        t_q = (t_span.to(torch.complex128) ** self.okhs.q).unsqueeze(1)
        lam = eig.unsqueeze(0)
        ML = _mittag_leffler_tensor(lam * t_q, self.okhs.q)
        
        X = c.unsqueeze(1) * Xi
        x_pred = ML @ X
        if return_tensor:
            return x_pred.real
        
        return x_pred.real.detach().cpu().numpy()

    def predict_with_diagnostics(
            self,
            initial_trajectory,
            t_span,
            stability_threshold=None,
            prediction_mode_selection_policy="all_stable",
            max_prediction_modes=None,
            min_prediction_modes=4,
    ):
        state = self._prepare_prediction_state(
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            stability_threshold=stability_threshold,
            prediction_mode_selection_policy=prediction_mode_selection_policy,
            max_prediction_modes=max_prediction_modes,
            min_prediction_modes=min_prediction_modes,
        )
        predicted = state["predicted"].detach().cpu().numpy()
        diagnostics = self._build_prediction_diagnostics(state)
        return predicted, diagnostics
    

    def plot_predict(self, initial_trajectory, t_span, stability_threshold=0.05):
        warnings.warn(
            "plot_predict is deprecated; use plot_forecast_diagnostics instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_forecast_diagnostics(
            self,
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            stability_threshold=stability_threshold,
        )


def plot_forecast_diagnostics(fdmd, initial_trajectory, t_span, stability_threshold=None):
    import matplotlib.pyplot as plt
    tab10 = plt.get_cmap("tab10")
    tab20 = plt.get_cmap("tab20")

    initial_trajectory = fdmd._normalize_state_trajectory(initial_trajectory)
    device = initial_trajectory.device
    t_span_tensor = torch.tensor(t_span, dtype=torch.float64, device=device)

    eig_full = fdmd.liouville_operator.eigenvalues_
    xi_full = fdmd.modes_
    stable_mask = select_stable_modes(
        eig_full,
        stability_policy=fdmd.stability_policy,
        stability_threshold=stability_threshold,
    )
    eig = eig_full[stable_mask]
    xi = xi_full[stable_mask]
    n_modes = len(eig)

    coefficients = fdmd.fit_initial_coefficients(initial_trajectory, Xi=xi, eig=eig)
    t_q = (t_span_tensor.to(torch.complex128) ** fdmd.okhs.q).unsqueeze(1)
    lam = eig.unsqueeze(0)
    mittag_tensor = _mittag_leffler_tensor(lam * t_q, fdmd.okhs.q)
    predicted_tensor = (mittag_tensor @ (coefficients.unsqueeze(1) * xi)).real
    
    # Перенос данных в NumPy для Matplotlib
    t_span = t_span_tensor.cpu().numpy()
    eig = eig.cpu().detach().numpy()
    xi = xi.cpu().detach().numpy()
    coefficients = coefficients.cpu().detach().numpy()
    mittag = mittag_tensor.cpu().detach().numpy()
    predicted = predicted_tensor.cpu().detach().numpy()

    abs_c = np.abs(coefficients)
    top_k = min(15, n_modes)
    top_idx = np.argsort(abs_c)[::-1][:top_k]
    other_idx = np.setdiff1d(np.arange(n_modes), top_idx)
    colors = [tab10(index) for index in range(top_k)] if top_k <= 10 else [tab20(index) for index in range(top_k)]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Анализ компонентов модели fDMD (OKHS)", fontsize=16, fontweight="bold")

    axis_ml = axes[0, 0]
    for color_index, mode_index in enumerate(top_idx):
        axis_ml.plot(
            t_span,
            np.real(mittag[:, mode_index]),
            color=colors[color_index],
            linewidth=1.5,
            label=f"j={mode_index}  λ={eig[mode_index]:.3f}",
        )
    axis_ml.set_xlabel("Время t")
    axis_ml.set_ylabel("Re(E_q(λ t^q))")
    axis_ml.set_title(f"Mittag-Leffler функции (топ-{top_k})")
    axis_ml.legend(loc="best", fontsize=8, ncol=2)
    axis_ml.grid(True, alpha=0.3)

    axis_c = axes[0, 1]
    positions = np.arange(top_k)
    bars_c = axis_c.bar(positions, abs_c[top_idx], color=colors, edgecolor="black", linewidth=0.5)
    axis_c.set_xlabel("Индекс моды (отсортированы по |c|)")
    axis_c.set_ylabel("|c_j|")
    axis_c.set_title("Коэффициенты разложения")
    axis_c.set_xticks(positions)
    axis_c.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_c, abs_c[top_idx]):
        axis_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center",
                    va="bottom", fontsize=7)

    axis_xi = axes[1, 0]
    abs_xi0 = np.abs(xi[top_idx, 0])
    bars_xi = axis_xi.bar(positions, abs_xi0, color=colors, edgecolor="black", linewidth=0.5)
    axis_xi.set_xlabel("Индекс моды (отсортированы по |c|)")
    axis_xi.set_ylabel("|Xi[j,0]|")
    axis_xi.set_title("Первая компонента мод Лиувилля")
    axis_xi.set_xticks(positions)
    axis_xi.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_xi, abs_xi0):
        axis_xi.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center",
                     va="bottom", fontsize=7)

    axis_eig = axes[1, 1]
    if len(other_idx) > 0:
        axis_eig.scatter(np.real(eig[other_idx]), np.imag(eig[other_idx]), c="gray", alpha=0.5, s=20,
                         label="прочие моды")
    for color_index, mode_index in enumerate(top_idx):
        axis_eig.scatter(
            np.real(eig[mode_index]),
            np.imag(eig[mode_index]),
            color=colors[color_index],
            s=80,
            edgecolor="black",
            linewidth=0.5,
        )
        axis_eig.annotate(str(mode_index), (np.real(eig[mode_index]), np.imag(eig[mode_index])),
                          textcoords="offset points", xytext=(5, 5), fontsize=6)
    axis_eig.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axis_eig.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    axis_eig.set_xlabel("Re(λ)")
    axis_eig.set_ylabel("Im(λ)")
    axis_eig.set_title("Собственные значения λ_j")
    axis_eig.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if fdmd.verbose:
        unstable = np.where(np.real(eig) > 0.1)[0]
        print("\n=== Диагностика ===")
        print(f"Всего мод: {n_modes}, показано топ-{top_k} по |c|")
        print(f"Диапазон Re(λ): [{np.min(np.real(eig)):.3f}, {np.max(np.real(eig)):.3f}]")
        print(f"Диапазон Im(λ): [{np.min(np.imag(eig)):.3f}, {np.max(np.imag(eig)):.3f}]")
        if len(unstable) > 0:
            print("Моды с положительной действительной частью (возможный рост):")
            for mode_index in unstable[:5]:
                print(f"  j={mode_index}, λ={eig[mode_index]:.3f}, |c|={abs_c[mode_index]:.3f}")

    return predicted
