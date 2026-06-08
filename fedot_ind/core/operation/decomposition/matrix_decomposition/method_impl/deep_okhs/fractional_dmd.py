from __future__ import annotations

import math
import warnings

import numpy as np
from pymonad import state
import torch
from scipy.special import roots_jacobi
from sklearn.base import BaseEstimator, RegressorMixin

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.matrix_utils import (
    validate_initial_coefficient_feasibility,
    select_stable_modes,
)
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.mittag_leffler import \
    _mittag_leffler_tensor
from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.deep_okhs.policies import (
    RegularizationPolicy,
    StabilityPolicy,
)


class FractionalDMD(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            liouville_operator,
            n_quad_points=20,
            regularization=1e-8,
            regularization_policy=None,
            stability_policy=None,
            verbose=False,
            device='cpu',
    ):
        self.liouville_operator = liouville_operator
        self.okhs = liouville_operator.okhs
        self.n_quad_points = n_quad_points
        self.regularization = regularization
        self.regularization_policy = regularization_policy or RegularizationPolicy()
        self.stability_policy = stability_policy or StabilityPolicy()
        self.verbose = verbose

        requested_device = torch.device(device)
        if requested_device.type == 'cuda' and not torch.cuda.is_available():
            if self.verbose:
                print('Warning: CUDA is not available. Falling back to CPU.')
            self.device = torch.device('cpu')
        else:
            self.device = requested_device

        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128

        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.modes_ = None
        self.initial_coefficients_ = None
        self._quad_cache = None

    def _to_tensor(self, value, *, dtype=None, device=None):
        target_device = device or self.device
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=target_device)
            return tensor.to(dtype=dtype) if dtype is not None else tensor
        return torch.as_tensor(value, dtype=dtype, device=target_device)

    def _to_real_tensor(self, value, *, device=None):
        return self._to_tensor(value, dtype=self.real_dtype, device=device)

    def _to_complex_tensor(self, value, *, device=None):
        return self._to_tensor(value, dtype=self.complex_dtype, device=device)

    def _to_index_tensor(self, value, *, device=None):
        return self._to_tensor(value, dtype=torch.long, device=device)

    def _to_bool_tensor(self, value, *, device=None):
        return self._to_tensor(value, dtype=torch.bool, device=device)

    def _coerce_prediction_inputs(self, initial_trajectory, t_span, eig=None, xi=None):
        normalized_trajectory = self._normalize_state_trajectory(initial_trajectory)
        normalized_t_span = self._coerce_time_grid(t_span)
        normalized_eig = None if eig is None else self._to_complex_tensor(eig)
        normalized_xi = None if xi is None else self._to_complex_tensor(xi)
        return normalized_trajectory, normalized_t_span, normalized_eig, normalized_xi

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _normalize_state_trajectory(self, trajectory):
        normalized = self._to_real_tensor(trajectory)
        if normalized.ndim == 1:
            return normalized.view(-1, 1)
        return normalized

    def _coerce_time_grid(self, t_span):
        return self._to_real_tensor(t_span).view(-1)

    def _get_eigenvalues_tensor(self):
        source = self.eigenvalues_
        if source is None:
            source = self.liouville_operator.eigenvalues_
        tensor = self._to_complex_tensor(source)
        self.eigenvalues_ = tensor
        return tensor

    def _get_eigenvectors_tensor(self):
        source = self.eigenvectors_
        if source is None:
            source = self.liouville_operator.eigenvectors_
        tensor = self._to_complex_tensor(source)
        self.eigenvectors_ = tensor
        return tensor

    def _align_modes_tensor(self, modes, *, eigenvalues=None):
        aligned = self._to_complex_tensor(modes)
        if aligned.ndim == 1:
            aligned = aligned.view(-1, 1)
        if aligned.ndim != 2:
            raise ValueError(
                f'FractionalDMD modes must be a 2D tensor, got shape {tuple(aligned.shape)}.'
            )

        reference = eigenvalues
        if reference is None:
            if self.eigenvalues_ is not None:
                reference = self.eigenvalues_
            else:
                reference = getattr(self.liouville_operator, 'eigenvalues_', None)

        if reference is None:
            return aligned

        n_modes = int(self._to_complex_tensor(reference).numel())
        if aligned.shape[0] == n_modes:
            return aligned
        if aligned.shape[1] == n_modes:
            return aligned.transpose(0, 1).contiguous()

        raise ValueError(
            'FractionalDMD modes are not aligned with eigenvalues: '
            f'eigenvalue_count={n_modes}, modes_shape={tuple(aligned.shape)}.'
        )

    def _get_modes_tensor(self, *, eigenvalues=None):
        if self.modes_ is None:
            raise ValueError('FractionalDMD modes are not initialized. Call fit first.')
        self.modes_ = self._align_modes_tensor(self.modes_, eigenvalues=eigenvalues)
        return self.modes_

    @staticmethod
    def _build_mode_groups(eigenvalues, tolerance=1e-8):
        groups = []
        used = set()
        eig_tensor = torch.as_tensor(eigenvalues, dtype=torch.complex128).detach().cpu()
        for index, eigenvalue_tensor in enumerate(eig_tensor):
            if index in used:
                continue
            eigenvalue = complex(eigenvalue_tensor.item())
            if abs(eigenvalue.imag) <= tolerance:
                groups.append((index,))
                used.add(index)
                continue
            conjugate_index = None
            for candidate_index in range(index + 1, len(eig_tensor)):
                if candidate_index in used:
                    continue
                candidate_value = complex(eig_tensor[candidate_index].item())
                if abs(candidate_value - np.conj(eigenvalue)) <= max(tolerance, 1e-6 * abs(eigenvalue)):
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
    def _resolve_prediction_mode_budget(
            available_equations,
            horizon,
            max_prediction_modes=None,
            min_prediction_modes=4):
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
            prediction_mode_selection_policy='all_stable',
            max_prediction_modes=None,
            min_prediction_modes=4,
    ):
        initial_trajectory, t_span, eig, xi = self._coerce_prediction_inputs(
            initial_trajectory,
            t_span,
            eig=eig,
            xi=xi,
        )
        stable_indices = self._to_index_tensor(stable_indices, device=self.device)

        policy = str(prediction_mode_selection_policy or 'all_stable').lower()
        horizon = int(t_span.numel())
        all_indices = torch.arange(len(eig), device=self.device, dtype=torch.long)
        default_budget = min(len(eig), max(1, int(available_equations)))

        if policy in {'all_stable', 'none'}:
            selected = all_indices[:default_budget]
            return {
                'eig': eig.index_select(0, selected),
                'xi': xi.index_select(0, selected),
                'selected_local_indices': selected,
                'selected_global_indices': stable_indices.index_select(0, selected),
                'policy': policy,
                'prediction_mode_budget': default_budget,
                'preselection_count': int(selected.numel()),
                'group_count': int(selected.numel()),
                'score_table': [],
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
                'eig': eig,
                'xi': xi,
                'selected_local_indices': selected,
                'selected_global_indices': stable_indices,
                'policy': policy,
                'prediction_mode_budget': budget,
                'preselection_count': int(selected.numel()),
                'group_count': int(selected.numel()),
                'score_table': [],
            }

        groups = self._build_mode_groups(eig)
        group_descriptors = []
        for group in groups:
            group_tensor = self._to_index_tensor(group)
            group_eig = eig.index_select(0, group_tensor)
            group_xi = xi.index_select(0, group_tensor)
            norm_sum = torch.sum(torch.linalg.norm(group_xi, dim=1))
            imag_sum = torch.sum(torch.abs(group_eig.imag))
            real_sum = torch.sum(torch.clamp(group_eig.real, min=0.0))
            preliminary_score = norm_sum * (1.0 + 0.5 * imag_sum + 0.25 * real_sum)
            group_descriptors.append(
                {
                    'indices': tuple(int(item) for item in group),
                    'size': int(len(group)),
                    'preliminary_score': float(preliminary_score.item()),
                }
            )

        preselection_budget = min(len(eig), max(budget, min(int(available_equations), int(max(budget * 3, budget)))))
        preliminary_sorted = sorted(group_descriptors, key=lambda item: item['preliminary_score'], reverse=True)

        preselected_local_indices_list = []
        preselected_groups = []
        for descriptor in preliminary_sorted:
            if len(preselected_local_indices_list) + descriptor['size'] > preselection_budget and preselected_groups:
                continue
            preselected_groups.append(descriptor)
            preselected_local_indices_list.extend(descriptor['indices'])
            if len(preselected_local_indices_list) >= preselection_budget:
                break

        preselected_local_indices = self._to_index_tensor(sorted(set(preselected_local_indices_list)))
        preselected_eig = eig.index_select(0, preselected_local_indices)
        preselected_xi = xi.index_select(0, preselected_local_indices)

        coefficients = self.fit_initial_coefficients(
            initial_trajectory,
            eig=preselected_eig,
            Xi=preselected_xi,
            store_result=False,
        )
        q_value = self.okhs.q
        tail_points = min(max(4, horizon // 2), int(initial_trajectory.shape[0]))
        
        # Получаем физическую сетку через менеджер и нормализуем
        tail_grid_phys = self.okhs.time_manager.get_physical_grid(initial_trajectory)[-tail_points:]
        tail_grid_norm = self.okhs.time_manager.normalize(tail_grid_phys)
        
        future_grid_norm = self.okhs.time_manager.normalize(t_span)

        # Используем нормализованные сетки в функции Миттаг-Леффлера
        tail_ml = _mittag_leffler_tensor(
            preselected_eig.unsqueeze(0) * (tail_grid_norm.to(self.complex_dtype).unsqueeze(1) ** q_value),
            q_value,
        )
        future_ml = _mittag_leffler_tensor(
            preselected_eig.unsqueeze(0) * (future_grid_norm.to(self.complex_dtype).unsqueeze(1) ** q_value),
            q_value,
        )

        local_index_map = {
            int(global_idx): pos
            for pos, global_idx in enumerate(preselected_local_indices.detach().cpu().tolist())
        }
        scored_groups = []
        for descriptor in preselected_groups:
            local_positions = self._to_index_tensor([local_index_map[idx] for idx in descriptor['indices']])
            local_coefficients = coefficients.index_select(0, local_positions)
            local_modes = preselected_xi.index_select(0, local_positions)
            local_eig = preselected_eig.index_select(0, local_positions)

            tail_component = (
                tail_ml.index_select(1, local_positions).unsqueeze(2)
                * local_coefficients.unsqueeze(0).unsqueeze(2)
                * local_modes.unsqueeze(0)
            ).real
            future_component = (
                future_ml.index_select(1, local_positions).unsqueeze(2)
                * local_coefficients.unsqueeze(0).unsqueeze(2)
                * local_modes.unsqueeze(0)
            ).real

            tail_energy = float(torch.mean(torch.abs(tail_component)).item())
            future_energy = float(torch.mean(torch.abs(future_component)).item())
            future_variability = float(torch.mean(torch.std(future_component, dim=0)).item())
            oscillation_score = float(torch.sum(torch.abs(local_eig.imag)).item())
            refined_score = 0.6 * tail_energy + 0.4 * future_variability + 0.2 * future_energy + 0.1 * oscillation_score
            scored_groups.append(
                {
                    'indices': descriptor['indices'],
                    'size': descriptor['size'],
                    'tail_energy': tail_energy,
                    'future_energy': future_energy,
                    'future_variability': future_variability,
                    'oscillation_score': oscillation_score,
                    'refined_score': refined_score,
                }
            )

        scored_groups.sort(key=lambda item: item['refined_score'], reverse=True)
        selected_local_indices_list = []
        for descriptor in scored_groups:
            if len(selected_local_indices_list) + descriptor['size'] > budget and selected_local_indices_list:
                continue
            selected_local_indices_list.extend(descriptor['indices'])
            if len(selected_local_indices_list) >= budget:
                break

        min_required_modes = min(int(min_prediction_modes), int(preselected_local_indices.numel()))
        if len(selected_local_indices_list) < min_required_modes:
            for descriptor in scored_groups:
                for index in descriptor['indices']:
                    if index not in selected_local_indices_list:
                        selected_local_indices_list.append(index)
                if len(selected_local_indices_list) >= min_required_modes:
                    break

        selected_local_indices = self._to_index_tensor(sorted(set(selected_local_indices_list)))
        return {
            'eig': eig.index_select(0, selected_local_indices),
            'xi': xi.index_select(0, selected_local_indices),
            'selected_local_indices': selected_local_indices,
            'selected_global_indices': stable_indices.index_select(0, selected_local_indices),
            'policy': policy,
            'prediction_mode_budget': budget,
            'preselection_count': int(preselected_local_indices.numel()),
            'group_count': int(len(groups)),
            'score_table': [
                {
                    'indices': list(item['indices']),
                    'refined_score': float(item['refined_score']),
                    'tail_energy': float(item['tail_energy']),
                    'future_energy': float(item['future_energy']),
                    'future_variability': float(item['future_variability']),
                    'oscillation_score': float(item['oscillation_score']),
                }
                for item in scored_groups[:10]
            ],
        }

    def _prepare_prediction_state(
            self,
            initial_trajectory,
            t_span,
            stability_threshold=None,
            prediction_mode_selection_policy='all_stable',
            max_prediction_modes=None,
            min_prediction_modes=4,
    ):
        initial_trajectory, t_span, _, _ = self._coerce_prediction_inputs(initial_trajectory, t_span)
        eig_full = self._get_eigenvalues_tensor()
        xi_full = self._get_modes_tensor(eigenvalues=eig_full)

        stable_mask = self._to_bool_tensor(
            select_stable_modes(
                eig_full,
                stability_policy=self.stability_policy,
                stability_threshold=stability_threshold,
            )
        )
        stable_indices = torch.where(stable_mask)[0]
        eig = eig_full.index_select(0, stable_indices)
        xi = xi_full.index_select(0, stable_indices)
        if eig.numel() == 0:
            raise ValueError('No stable modes remain after applying the stability policy.')

        available_equations = int(initial_trajectory.shape[0] * initial_trajectory.shape[1])
        selection = self._select_prediction_modes(
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            eig=eig,
            xi=xi,
            stable_indices=stable_indices,
            available_equations=available_equations,
            prediction_mode_selection_policy=prediction_mode_selection_policy,
            max_prediction_modes=max_prediction_modes,
            min_prediction_modes=min_prediction_modes,
        )

        eig = selection['eig']
        xi = selection['xi']
        selected_mode_indices = selection['selected_global_indices']
        prediction_mode_cap = int(selection['prediction_mode_budget'])
        prediction_mode_cap_applied = bool(int(stable_indices.numel()) > int(selected_mode_indices.numel()))

        coefficients = self.fit_initial_coefficients(initial_trajectory, Xi=xi, eig=eig)
        
        # Нормализация времени для функции Миттаг-Леффлера
        t_span_norm = self.okhs.time_manager.normalize(t_span)
        t_q = (t_span_norm.to(self.complex_dtype) ** self.okhs.q).unsqueeze(1)
        
        mittag = _mittag_leffler_tensor(eig.unsqueeze(0) * t_q, self.okhs.q)
        predicted = (mittag @ (coefficients.unsqueeze(1) * xi)).real.to(self.real_dtype)
        
        #Начало предсказания заполняем тем, что наблюдали
        predicted[:initial_trajectory.shape[0]] = initial_trajectory

        return {
            'initial_trajectory': initial_trajectory,
            't_span': t_span, # В state сохраняем исходное время, чтобы можно было строить графики
            'eig_full': eig_full,
            'stable_mask': stable_mask,
            'eig': eig,
            'xi': xi,
            'selected_mode_indices': selected_mode_indices,
            'available_equations': available_equations,
            'prediction_mode_cap': prediction_mode_cap,
            'prediction_mode_cap_applied': prediction_mode_cap_applied,
            'prediction_mode_selection_policy': selection['policy'],
            'prediction_preselection_count': selection['preselection_count'],
            'prediction_group_count': selection['group_count'],
            'prediction_mode_scores': selection['score_table'],
            'coefficients': coefficients,
            'predicted': predicted,
            'tensor_device': str(self.device),
            'tensor_dtype': str(self.real_dtype),
        }

    def _build_prediction_diagnostics(self, state):
        initial_trajectory = state['initial_trajectory']
        eig = state['eig']
        xi = state['xi']
        coefficients = state['coefficients']
        predicted = state['predicted']
        stable_mask = state['stable_mask']
        selected_mode_indices = state['selected_mode_indices']

        # Получение физической сетки через менеджер и ее нормализация
        initial_grid_phys = self.okhs.time_manager.get_physical_grid(initial_trajectory)
        initial_grid_norm = self.okhs.time_manager.normalize(initial_grid_phys)
        
        initial_t_q = (initial_grid_norm.to(self.complex_dtype) ** self.okhs.q).unsqueeze(1)
        initial_mittag = _mittag_leffler_tensor(eig.unsqueeze(0) * initial_t_q, self.okhs.q)
        reconstruction = (initial_mittag @ (coefficients.unsqueeze(1) * xi)).real

        last_observed = initial_trajectory[-1].reshape(-1)
        first_prediction = predicted[0].reshape(-1) if len(predicted) else last_observed
        discontinuity = first_prediction - last_observed

        return {
            'n_total_modes': int(len(state['eig_full'])),
            'n_stable_modes': int(torch.sum(stable_mask).item()),
            'stable_mode_indices': torch.where(stable_mask)[0].tolist(),
            'n_selected_prediction_modes': int(len(eig)),
            'selected_prediction_mode_indices': selected_mode_indices.tolist(),
            'available_equations': int(state['available_equations']),
            'prediction_mode_cap': int(state['prediction_mode_cap']),
            'prediction_mode_cap_applied': bool(state['prediction_mode_cap_applied']),
            'prediction_mode_selection_policy': state['prediction_mode_selection_policy'],
            'prediction_preselection_count': int(state['prediction_preselection_count']),
            'prediction_group_count': int(state['prediction_group_count']),
            'prediction_mode_scores': state['prediction_mode_scores'],
            'stable_eigenvalues_real': eig.real.detach().cpu().tolist(),
            'stable_eigenvalues_imag': eig.imag.detach().cpu().tolist(),
            'mode_norms': torch.linalg.norm(xi, dim=1).detach().cpu().tolist() if len(xi) else [],
            'initial_coefficients_real': coefficients.real.detach().cpu().tolist(),
            'initial_coefficients_imag': coefficients.imag.detach().cpu().tolist(),
            'initial_reconstruction_rmse': float(
                torch.sqrt(torch.mean((reconstruction - initial_trajectory) ** 2)).item()),
            'initial_reconstruction_mae': float(torch.mean(torch.abs(reconstruction - initial_trajectory)).item()),
            'last_observed_value': last_observed.detach().cpu().tolist(),
            'first_prediction_value': first_prediction.detach().cpu().tolist(),
            'boundary_discontinuity': discontinuity.detach().cpu().tolist(),
            'boundary_discontinuity_abs_mean': float(torch.mean(torch.abs(discontinuity)).item()),
            'tensor_device': state.get('tensor_device', str(self.device)),
            'tensor_dtype': state.get('tensor_dtype', str(self.real_dtype)),
        }

    def _get_jacobi_rule(self):
        if self._quad_cache is None:
            nodes, weights = roots_jacobi(self.n_quad_points, self.okhs.q - 1, 0)
            self._quad_cache = (
                self._to_real_tensor(nodes),
                self._to_real_tensor(weights),
            )
        else:
            nodes, weights = self._quad_cache
            if nodes.device != self.device or weights.device != self.device:
                self._quad_cache = (nodes.to(self.device), weights.to(self.device))
        return self._quad_cache

    def compute_identity_projections(self, trajectories):
        n_traj = len(trajectories)
        n_features = self._normalize_state_trajectory(trajectories[0]).shape[1]
        Y = torch.zeros((n_traj, n_features), dtype=self.real_dtype, device=self.device)
        _, weights = self._get_jacobi_rule()

        for k in range(n_traj):
            traj = self._normalize_state_trajectory(trajectories[k])
            T_traj = self.okhs._get_trajectory_duration(traj)
            if T_traj <= 1e-14:
                continue
            vals = self.okhs._evaluate_trajectory_at_nodes(traj, T_traj)
            vals = self._to_real_tensor(vals)
            jacobian_factor = (T_traj / 2.0) ** self.okhs.q
            integral_sum = torch.einsum('k,kd->d', weights, vals)
            Y[k, :] = self.okhs.C_q * jacobian_factor * integral_sum
        return Y

    def compute_eigenfunction_projections(self, Y, V):
        return V.conj().T @ Y

    def solve_modes(self, W, B):
        regularization = max(self.regularization, self.regularization_policy.base_jitter)
        W_reg = W + regularization * torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
        try:
            return torch.linalg.solve(W_reg, B)
        except RuntimeError:
            return torch.linalg.pinv(W_reg) @ B

    def fit(self, trajectories=None):
        if trajectories is None:
            trajectories = self.okhs.train_trajectories_
        if self.liouville_operator.eigenvectors_ is None:
            raise ValueError('Liouville operator must be fitted.')

        self.eigenvalues_ = self._get_eigenvalues_tensor()
        self.eigenvectors_ = self._get_eigenvectors_tensor()
        V = self.eigenvectors_
        G = self._to_complex_tensor(self.okhs.gram_matrix_)
        Y = self.compute_identity_projections(trajectories).to(V.dtype)
        B = self.compute_eigenfunction_projections(Y, V)
        W = V.conj().T @ G @ V
        self.modes_ = self._align_modes_tensor(
            self.solve_modes(W, B),
            eigenvalues=self.eigenvalues_,
        ).to(device=self.device)
        return self

    def fit_initial_coefficients(self, initial_trajectory, eig=None, Xi=None, store_result=True):
        import math
        
        initial_trajectory = self._normalize_state_trajectory(initial_trajectory)
        K, n_features = initial_trajectory.shape

        eig = self._to_complex_tensor(eig if eig is not None else self._get_eigenvalues_tensor())
        Xi = self._align_modes_tensor(
            Xi if Xi is not None else self._get_modes_tensor(eigenvalues=eig),
            eigenvalues=eig,
        )
        n_modes = int(eig.shape[0])
        validate_initial_coefficient_feasibility(initial_trajectory, n_modes)

        tail_points = math.ceil(n_modes / n_features)
        tail_points = min(tail_points, K)
        initial_trajectory_tail = initial_trajectory[-tail_points:]

        # Физическая сетка запрашивается из менеджера и нормализуется 
        t_grid_full_phys = self.okhs.time_manager.get_physical_grid(initial_trajectory)
        t_grid_tail_phys = t_grid_full_phys[-tail_points:]
        t_grid_tail_norm = self.okhs.time_manager.normalize(t_grid_tail_phys)

        # Передаем НОРМАЛИЗОВАННОЕ время в функцию Миттаг-Леффлера
        ml_evals = _mittag_leffler_tensor(
            eig.unsqueeze(0) * (t_grid_tail_norm.to(self.complex_dtype).unsqueeze(1) ** self.okhs.q),
            self.okhs.q,
        )
        
        A_blocks = ml_evals.unsqueeze(2) * Xi.unsqueeze(0)
        
        A = A_blocks.transpose(1, 2).reshape(tail_points * n_features, n_modes)
        b = initial_trajectory_tail.reshape(-1).to(self.complex_dtype)

        alpha = max(self.regularization, self.regularization_policy.base_jitter)
        A_stack = A.conj().T @ A
        reg_matrix = A_stack + alpha * torch.eye(n_modes, dtype=A_stack.dtype, device=self.device)
        b_stack = A.conj().T @ b
        
        try:
            c = torch.linalg.solve(reg_matrix, b_stack)
        except RuntimeError:
            c = torch.linalg.pinv(reg_matrix) @ b_stack

        # try:
        #     c = torch.linalg.lstsq(A, b).solution
        # except RuntimeError:
        #     c = torch.linalg.pinv(A) @ b

        if store_result:
            self.initial_coefficients_ = c
        return c

    def predict(self, initial_trajectory, t_span, stability_threshold=None, return_tensor=False):
        state = self._prepare_prediction_state(
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            stability_threshold=stability_threshold,
            prediction_mode_selection_policy='all_stable',
            max_prediction_modes=None,
            min_prediction_modes=1,
        )
        predicted = state['predicted']

        # try:
        #     import matplotlib.pyplot as plt
        #     coefficients = state['coefficients']
        #     eig = state['eig']
        #     xi = state['xi']
            
        #     # 1. Извлекаем физические сетки времени (ось X) и строго делаем их 1D-массивами
        #     initial_grid_phys = self.okhs.time_manager.get_physical_grid(initial_trajectory)
        #     t_initial_np = initial_grid_phys.detach().cpu().numpy().flatten()
        #     t_forecast_np = self._coerce_time_grid(t_span).detach().cpu().numpy().flatten()
            
        #     # 2. Строим реконструкцию: обязательно переносим нормализованное время на устройство ядра
        #     initial_grid_norm = self.okhs.time_manager.normalize(initial_grid_phys).to(self.device)
        #     initial_t_q = (initial_grid_norm.to(self.complex_dtype) ** self.okhs.q).unsqueeze(1)
            
        #     initial_mittag = _mittag_leffler_tensor(eig.unsqueeze(0) * initial_t_q, self.okhs.q)
        #     reconstruction = (initial_mittag @ (coefficients.unsqueeze(1) * xi)).real.detach().cpu().numpy()
            
        #     # 3. Готовим остальные массивы 
        #     initial_traj_np = self._normalize_state_trajectory(initial_trajectory).detach().cpu().numpy()
        #     predicted_np = predicted.detach().cpu().numpy()

        #     # Защита от потери размерности фазового пространства (на случай если массивы стали 1D)
        #     if initial_traj_np.ndim == 1: initial_traj_np = initial_traj_np[:, None]
        #     if reconstruction.ndim == 1: reconstruction = reconstruction[:, None]
        #     if predicted_np.ndim == 1: predicted_np = predicted_np[:, None]

        #     plt.figure(figsize=(10, 5))
        #     dim_idx = 0  # Рисуем только 0-ю компоненту
            
        #     # Находим середину исторического куска
        #     length_initial = len(t_initial_np) // 2
            
        #     # Рисуем ИСТИННУЮ ИСТОРИЮ (только вторую половину)
        #     plt.plot(t_initial_np[length_initial:], initial_traj_np[length_initial:, dim_idx], 
        #              'k-', linewidth=2, alpha=0.6, label='Истинная история')
            
        #     # Рисуем РЕКОНСТРУКЦИЮ (только вторую половину)
        #     plt.plot(t_initial_np[length_initial:], reconstruction[length_initial:, dim_idx], 
        #              'b--', linewidth=2, label='Реконструкция fDMD (Fit)')
            
        #     # Рисуем ПРОГНОЗ целиком (он начинается после истории)
        #     plt.plot(t_forecast_np, predicted_np[:, dim_idx], 
        #              'r-', linewidth=2, label='Прогноз (Forecast)')
            
        #     plt.axvline(x=t_initial_np[-1], color='gray', linestyle=':', label='Граница прогноза')
            
        #     plt.title("Отладка скачка прогноза fDMD (Увеличенный масштаб у границы)")
        #     plt.xlabel("Физическое время")
        #     plt.ylabel("Значение ряда")
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #     plt.tight_layout()
        #     plt.show()
            
        #     # 5. Вывод текстовой диагностики
        #     diagnostics = self._build_prediction_diagnostics(state)
        #     print(f"\n[DEBUG] Истинное последнее значение:    {diagnostics['last_observed_value']}")
        #     print(f"[DEBUG] Первое значение прогноза:      {diagnostics['first_prediction_value']}")
        #     print(f"[DEBUG] Абсолютный разрыв:      {diagnostics['boundary_discontinuity_abs_mean']:.6f}\n")
        #     print(f"[DEBUG] Количество стабильных мод:      {diagnostics['n_stable_modes']} из {diagnostics['n_total_modes']}")
        # except Exception as e:
        #     # Если что-то пойдет не так, мы просто напечатаем ошибку и продолжим прогноз
        #     print(f"\n[DEBUG ERROR] Ошибка при построении графика отладки: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     print("--------------------------------------------------\n")
        # ===================================================
        # predicted = predicted * self.okhs.max_elem
        if return_tensor:
            return predicted
        return predicted.detach().cpu().numpy()

    def predict_with_diagnostics(
            self,
            initial_trajectory,
            t_span,
            stability_threshold=None,
            prediction_mode_selection_policy='all_stable',
            max_prediction_modes=None,
            min_prediction_modes=4,
            return_tensor=False,
    ):
        state = self._prepare_prediction_state(
            initial_trajectory=initial_trajectory,
            t_span=t_span,
            stability_threshold=stability_threshold,
            prediction_mode_selection_policy=prediction_mode_selection_policy,
            max_prediction_modes=max_prediction_modes,
            min_prediction_modes=min_prediction_modes,
        )
        diagnostics = self._build_prediction_diagnostics(state)
        if return_tensor:
            return state['predicted'], diagnostics
        return state['predicted'].detach().cpu().numpy(), diagnostics

    def plot_predict(self, initial_trajectory, t_span, stability_threshold=0.05):
        warnings.warn(
            'plot_predict is deprecated; use plot_forecast_diagnostics instead.',
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

    tab10 = plt.get_cmap('tab10')
    tab20 = plt.get_cmap('tab20')

    initial_trajectory = fdmd._normalize_state_trajectory(initial_trajectory)
    t_span_tensor = fdmd._coerce_time_grid(t_span)
    
    # Нормализация t_span для корректного вычисления 
    t_span_norm = fdmd.okhs.time_manager.normalize(t_span_tensor)

    eig_full = fdmd._get_eigenvalues_tensor()
    xi_full = fdmd._get_modes_tensor(eigenvalues=eig_full)

    stable_mask = fdmd._to_bool_tensor(
        select_stable_modes(
            eig_full,
            stability_policy=fdmd.stability_policy,
            stability_threshold=stability_threshold,
        )
    )
    stable_indices = torch.where(stable_mask)[0]
    eig = eig_full.index_select(0, stable_indices)
    xi = xi_full.index_select(0, stable_indices)
    n_modes = len(eig)

    coefficients = fdmd.fit_initial_coefficients(initial_trajectory, Xi=xi, eig=eig)
    
    # Используем нормализованное время t_span_norm в функции Миттаг-Леффлера
    t_q = (t_span_norm.to(fdmd.complex_dtype) ** fdmd.okhs.q).unsqueeze(1)
    mittag_tensor = _mittag_leffler_tensor(eig.unsqueeze(0) * t_q, fdmd.okhs.q)
    predicted_tensor = (mittag_tensor @ (coefficients.unsqueeze(1) * xi)).real

    # На графиках по оси Х оставляем физическое время t_span_np
    t_span_np = t_span_tensor.detach().cpu().numpy()
    eig_np = eig.detach().cpu().numpy()
    xi_np = xi.detach().cpu().numpy()
    coefficients_np = coefficients.detach().cpu().numpy()
    mittag_np = mittag_tensor.detach().cpu().numpy()
    predicted = predicted_tensor.detach().cpu().numpy()

    abs_c = np.abs(coefficients_np)
    top_k = min(15, n_modes)
    top_idx = np.argsort(abs_c)[::-1][:top_k]
    other_idx = np.setdiff1d(np.arange(n_modes), top_idx)
    colors = [tab10(index) for index in range(top_k)] if top_k <= 10 else [tab20(index) for index in range(top_k)]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    figure.suptitle('Fractional DMD (OKHS) Diagnostics', fontsize=16, fontweight='bold')

    axis_ml = axes[0, 0]
    for color_index, mode_index in enumerate(top_idx):
        axis_ml.plot(
            t_span_np,
            np.real(mittag_np[:, mode_index]),
            color=colors[color_index],
            linewidth=1.5,
            label=f'j={mode_index}, \u03bb={eig_np[mode_index]:.3f}'
        )
    axis_ml.set_xlabel('Time (t)')
    axis_ml.set_ylabel('Re(E_q(\u03bb \u03c4^q))')
    axis_ml.set_title(f'Mittag-Leffler Dynamics (Top {top_k})')
    axis_ml.legend(loc='best', fontsize=8, ncol=2)
    axis_ml.grid(True, alpha=0.3)

    axis_c = axes[0, 1]
    positions = np.arange(top_k)
    bars_c = axis_c.bar(positions, abs_c[top_idx], color=colors, edgecolor='black', linewidth=0.5)
    axis_c.set_xlabel('Mode Index (sorted by |c|)')
    axis_c.set_ylabel('|c_j|')
    axis_c.set_title('Initial Coefficients Amplitude')
    axis_c.set_xticks(positions)
    axis_c.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_c, abs_c[top_idx]):
        axis_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.3f}', ha='center',
                    va='bottom', fontsize=7)

    axis_xi = axes[1, 0]
    abs_xi0 = np.abs(xi_np[top_idx, 0])
    bars_xi = axis_xi.bar(positions, abs_xi0, color=colors, edgecolor='black', linewidth=0.5)
    axis_xi.set_xlabel('Mode Index (sorted by |c|)')
    axis_xi.set_ylabel('|\u03be_{j,0}|')
    axis_xi.set_title('Spatial Mode Amplitudes')
    axis_xi.set_xticks(positions)
    axis_xi.set_xticklabels([str(index) for index in top_idx], rotation=45)
    for bar, value in zip(bars_xi, abs_xi0):
        axis_xi.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.3f}', ha='center',
                     va='bottom', fontsize=7)

    axis_eig = axes[1, 1]
    if len(other_idx) > 0:
        axis_eig.scatter(np.real(eig_np[other_idx]), np.imag(eig_np[other_idx]), c='gray', alpha=0.5, s=20,
                         label='Other modes')
    for color_index, mode_index in enumerate(top_idx):
        axis_eig.scatter(
            np.real(eig_np[mode_index]),
            np.imag(eig_np[mode_index]),
            color=colors[color_index],
            s=80,
            edgecolor='black',
            linewidth=0.5,
        )
        axis_eig.annotate(str(mode_index), (np.real(eig_np[mode_index]), np.imag(eig_np[mode_index])),
                          textcoords='offset points', xytext=(5, 5), fontsize=6)
    axis_eig.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axis_eig.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    axis_eig.set_xlabel('Re(\u03bb)')
    axis_eig.set_ylabel('Im(\u03bb)')
    axis_eig.set_title('Eigenvalues of the fDMD Operator')
    axis_eig.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if fdmd.verbose:
        unstable = np.where(np.real(eig_np) > 0.1)[0]
        print('\n=== Prediction diagnostics ===')
        print(f'Total modes: {n_modes}, Showing top {top_k} by |c|')
        print(f'Re(\u03bb): [{np.min(np.real(eig_np)):.3f}, {np.max(np.real(eig_np)):.3f}]')
        print(f'Im(\u03bb): [{np.min(np.imag(eig_np)):.3f}, {np.max(np.imag(eig_np)):.3f}]')
        if len(unstable) > 0:
            print('Unstable modes (real parts > 0.1):')
            for mode_index in unstable[:5]:
                print(f'  j={mode_index}, \u03bb={eig_np[mode_index]:.3f}, |c|={abs_c[mode_index]:.3f}')

    return predicted