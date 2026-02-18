import numpy as np
from sklearn.base import TransformerMixin
from scipy.integrate import quad
from scipy.special import gamma
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import eig

# from ..core.kernels import OccupationKernel
# from ....transformation.representation.kernel.kernels import OccupationKernel

class OKHSTransformer(TransformerMixin):
    """
    Трансформер для OKHS признаков с численным интегрированием.
    
    Матрица Грама вычисляется согласно формуле (7) из статьи Rosenfeld et al.:
    
    G_{i,j} = C_q^2 * ∫∫_{[0,T]²} (T - τ)^{q-1} (T - t)^{q-1} K(ξ_j(t), ξ_i(τ)) dt dτ
    
    где:
    - C_q = Γ(q) / Γ²(q)
    - K — ядро из базового RKHS (e.g., RBF)
    - ξ_i, ξ_j — траектории
    - q — порядок системы (0 < q ≤ 1)
    """

    def __init__(self, kernel, q=0.7, eigenvalue_threshold=1e-6, 
                 integration_method='quad', n_quad_points=50):
        """
        Параметры:
        -----------
        kernel : KernelBase
            Ядро из класса RBFKernel или другого наследника KernelBase.
            Это базовое ядро RKHS K(x, y).
        
        q : float (0 < q ≤ 1)
            Порядок дробной производной системы. По умолчанию 0.7.
        
        n_components : int или None
            Количество компонент для сохранения. Если None, сохраняются все.
        
        eigenvalue_threshold : float
            Порог для фильтрации малых собственных значений при стабилизации
            обращения матрицы Грама.
        
        integration_method : str
            'quad' — scipy.integrate.quad
            'gaussian' — гауссовы квадратуры
        
        n_quad_points : int
            Количество точек квадратуры для метода 'gaussian'.
        
        quad_limit : int
            Максимальное число разбиений для метода 'quad'.
        """
        self.kernel = kernel
        self.q = q
        self.eigenvalue_threshold = eigenvalue_threshold
        self.integration_method = integration_method
        self.n_quad_points = n_quad_points
        self.quad_limit = 100 
        self.C_q = 1.0 / gamma(q)
        
    def _get_trajectory_duration(self, trajectory):
        """
        Получить длительность траектории (T - нормировано на [0,1]).
        
        trajectory : array-like, shape (n_steps, n_features)
            Дискретная траектория
        
        Returns
        -------
        T : float
            Нормированное время конца траектории (обычно 1.0 для [0,T])
        """
        return 1.0  # Нормируем на единичный интервал [0, 1]

    def _evaluate_trajectory_at_time(self, trajectory, t):
        """
        Интерполировать траекторию в момент времени t.
        
        trajectory : array-like, shape (n_steps, n_features)
        t : float
            Время в [0, 1]
        
        Returns
        -------
        value : array, shape (n_features,)
        """
        n_steps = len(trajectory)
        t_normalized = t * (n_steps - 1)  # Масштабируем на [0, n_steps-1]
        
        idx = int(np.floor(t_normalized))
        idx = np.clip(idx, 0, n_steps - 2)
        
        alpha = t_normalized - idx  # Коэффициент интерполяции
        
        # Линейная интерполяция
        value = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
        return value

    def _compute_gram_entry_quad(self, trajectory_i, trajectory_j):
        """
        Вычислить один элемент матрицы Грама используя численное интегрирование
        (scipy.integrate.quad).
        
        G_{i,j} = C_q^2 ∫∫ (T-τ)^{q-1} (T-t)^{q-1} K(ξ_j(t), ξ_i(τ)) dt dτ
        
        trajectory_i, trajectory_j : array-like, shape (n_steps, n_features)
        
        Returns
        -------
        gram_entry : float
        """
        T = self._get_trajectory_duration(trajectory_i)

        def kernel_weighted_inner(tau):
            """Интеграл по t для фиксированного τ"""
            def integrand(t):
                xi_i_tau = self._evaluate_trajectory_at_time(trajectory_i, tau)
                xi_j_t = self._evaluate_trajectory_at_time(trajectory_j, t)
                
                # Ядро между траекториями в моменты t и τ
                K_val = self.kernel._compute_single_kernel(xi_j_t, xi_i_tau)
                
                # Весовая функция: (T - τ)^{q-1} (T - t)^{q-1}
                weight = (T - tau) ** (self.q - 1) * (T - t) ** (self.q - 1)
                
                return K_val * weight
            
            # Интегрируем по t
            result, _ = quad(integrand, 0, T, limit=self.quad_limit)
            return result
        
        # Интегрируем по τ
        gram_entry, _ = quad(kernel_weighted_inner, 0, T, limit=self.quad_limit)
        
        # Множитель C_q^2
        gram_entry *= (self.C_q ** 2)
        
        return gram_entry

    def _compute_gram_entry_gaussian(self, trajectory_i, trajectory_j):
        """
        Вычислить один элемент матрицы Грама используя гауссовы квадратуры
        (numpy.polynomial.legendre.leggauss).
        
        trajectory_i, trajectory_j : array-like, shape (n_steps, n_features)
        
        Returns
        -------
        gram_entry : float
        """
        from numpy.polynomial.legendre import leggauss
        
        T = self._get_trajectory_duration(trajectory_i)
        
        # Получаем узлы и веса гауссовой квадратуры на [-1, 1]
        nodes, weights = leggauss(self.n_quad_points)
        
        # Масштабируем на [0, T]
        nodes_tau = T * (nodes + 1) / 2
        weights_tau = weights * T / 2
        
        gram_entry = 0.0
        
        for tau_idx, (tau, w_tau) in enumerate(zip(nodes_tau, weights_tau)):
            # Вложенная квадратура по t
            nodes_t = T * (nodes + 1) / 2
            weights_t = weights * T / 2
            
            for t, w_t in zip(nodes_t, weights_t):
                xi_i_tau = self._evaluate_trajectory_at_time(trajectory_i, tau)
                xi_j_t = self._evaluate_trajectory_at_time(trajectory_j, t)
                
                # Ядро
                K_val = self.kernel._compute_single_kernel(xi_j_t, xi_i_tau)
                
                # Весовая функция
                weight = (T - tau) ** (self.q - 1) * (T - t) ** (self.q - 1)
                
                gram_entry += K_val * weight * w_tau * w_t
        
        # Множитель C_q^2
        gram_entry *= (self.C_q ** 2)
        
        return gram_entry

    def _compute_gram_matrix(self, trajectories):
        """
        Вычислить полную матрицу Грама для набора траекторий.
        
        trajectories : list of array-like
            Список траекторий, каждая формы (n_steps, n_features)
        
        Returns
        -------
        gram_matrix : array, shape (n_trajectories, n_trajectories)
        """
        n = len(trajectories)
        gram_matrix = np.zeros((n, n))
        
        # Выбираем метод интегрирования
        if self.integration_method == 'quad':
            compute_entry = self._compute_gram_entry_quad
        elif self.integration_method == 'gaussian':
            compute_entry = self._compute_gram_entry_gaussian
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        
        for i in range(n):
            for j in range(i, n):  # Используем симметрию
                gram_matrix[i, j] = compute_entry(trajectories[i], trajectories[j])
                if i != j:
                    gram_matrix[j, i] = gram_matrix[i, j]  # Симметричность
        
        return gram_matrix

    def fit(self, trajectories, y=None):
        """
        Обучение трансформера: вычисление матрицы Грама и сохранение траекторий.
        
        Parameters
        ----------
        trajectories : list of array-like
            Список обучающих траекторий
        y : ignored
            Параметр scikit-learn interface (игнорируется)
        
        Returns
        -------
        self : OKHSTransformer
        """
        self.trajectories_ = trajectories
        
        print(f"Computing Gram matrix with {self.integration_method} integration "
              f"(q={self.q})...")
        self.gram_matrix_ = self._compute_gram_matrix(trajectories)
        
        # Стабилизация: добавляем регуляризацию для плохо обусловленной матрицы
        cond_number = np.linalg.cond(self.gram_matrix_)
        if cond_number > 1e10:
            regularization = 1e-8 * np.eye(self.gram_matrix_.shape[0])
            self.gram_matrix_ += regularization
        
        print(f"Gram matrix computed. Condition number: {np.linalg.cond(self.gram_matrix_):.2e}")
        
        return self

    def transform(self, trajectories):
        """
        Преобразование траекторий в координаты OKHS.
        
        Решаем систему: G c^T = K_{test,train}^T
        где:
        - G — матрица Грама обучающих траекторий
        - K_{test,train}[i,j] — ядро между test[i] и train[j]
        - c[i, :] — координаты i-й тестовой траектории
        
        Parameters
        ----------
        trajectories : list of array-like
            Список тестовых траекторий
        
        Returns
        -------
        features : array, shape (n_test, n_train)
            Координаты траекторий в базисе occupation kernels
        """
        n_train = len(self.trajectories_)
        n_test = len(trajectories)
        
        # Матрица ядер между test и train: K[i, j] = K(test_i, train_j)
        kernel_matrix = np.zeros((n_test, n_train))
        
        print("Computing kernel matrix between test and train trajectories...")
        for i in range(n_test):
            for j in range(n_train):
                # Используем метод _compute_gram_entry для консистентности
                kernel_matrix[i, j] = self._compute_gram_entry_quad(
                    trajectories[i], self.trajectories_[j]
                )
        
        # Решаем систему: G^T c^T = K^T => c = (G^{-1} K^T)^T
        try:
            c = np.linalg.solve(self.gram_matrix_, kernel_matrix.T).T
        except np.linalg.LinAlgError as e:
            print(f"Warning: Gram matrix is singular. Using pseudo-inverse. Error: {e}")
            c = kernel_matrix @ np.linalg.pinv(self.gram_matrix_).T
        
        return c
    

class FractionalLiouvilleOperator(BaseEstimator):
    """
    Оператор Лиувилля дробного порядка для спектрального анализа динамических систем.
    
    Реализует конечномерное представление оператора P A_{f,q} P по формуле (6)
    из статьи Rosenfeld et al. (2022).
    """

    def __init__(self, okhs_transformer, integration_method='quad', 
                 n_gaussian_points=30, quad_max_limit=100):
        """
        Parameters
        ----------
        okhs_transformer : OKHSTransformer
            Обученный экземпляр OKHSTransformer, содержащий матрицу Грама G 
            и обучающие траектории.
        
        integration_method : str
            'quad' или 'gaussian'. Должен совпадать с методом в трансформере
            для консистентности, но может быть выбран отдельно.
            
        n_gaussian_points : int
            Число точек для гауссовой квадратуры.
            
        quad_max_limit : int
            Лимит разбиений для адаптивной квадратуры.
        """
        self.okhs = okhs_transformer
        self.integration_method = integration_method
        self.n_gaussian_points = n_gaussian_points
        self.quad_max_limit = quad_max_limit
        
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.liouville_matrix_ = None

    def _compute_liouville_entry_quad(self, traj_i, traj_j):
        """
        Вычисление элемента матрицы Лиувилля через формулу (6.3):
        <A* mu_i, mu_j> = C_q * ∫ (T-τ)^(q-1) * [K(ξ_j(τ), ξ_i(T)) - K(ξ_j(τ), ξ_i(0))] dτ
        
        Обратите внимание:
        traj_i - траектория, дающая граничные условия (ξ_i(T), ξ_i(0))
        traj_j - траектория под интегралом (ξ_j(τ))
        """
        T = 1.0
        q = self.okhs.q
        C_q = self.okhs.C_q
        
        # Граничные точки i-й траектории
        # ξ_i(T) - конец, ξ_i(0) - начало
        xi_i_T = traj_i[-1] 
        xi_i_0 = traj_i[0]
        
        # Функция ядра K(x, y)
        kernel_func = self.okhs.kernel._compute_single_kernel
        evaluate = self.okhs._evaluate_trajectory_at_time

        def integrand(tau):
            # Точка на j-й траектории
            xi_j_tau = evaluate(traj_j, tau)
            
            # Вес
            weight = (T - tau) ** (q - 1)
            
            # Разность ядер на концах: K(ξ_j(τ), ξ_i(T)) - K(ξ_j(τ), ξ_i(0))
            k_diff = kernel_func(xi_j_tau, xi_i_T) - kernel_func(xi_j_tau, xi_i_0)
            
            return weight * k_diff

        # Одинарный интеграл
        res, _ = quad(integrand, 0, T, limit=self.quad_max_limit)
        return res * C_q

    def _compute_liouville_entry_gaussian(self, traj_i, traj_j):
        """
        Вычисление интеграла через гауссовы квадратуры.
        """
        from numpy.polynomial.legendre import leggauss
        
        T = 1.0
        q = self.okhs.q
        C_q = self.okhs.C_q
        
        # Узлы и веса
        nodes, weights = leggauss(self.n_gaussian_points)
        nodes_scaled = T * (nodes + 1) / 2
        weights_scaled = weights * T / 2
        
        xi_i_T = traj_i[-1]
        xi_i_0 = traj_i[0]
        kernel_func = self.okhs.kernel._compute_single_kernel
        evaluate = self.okhs._evaluate_trajectory_at_time
        
        integral = 0.0
        for i, tau in enumerate(nodes_scaled):
            xi_j_tau = evaluate(traj_j, tau)
            w_tau = weights_scaled[i]
            
            weight_func = (T - tau) ** (q - 1)
            k_diff = kernel_func(xi_j_tau, xi_i_T) - kernel_func(xi_j_tau, xi_i_0)
            
            integral += k_diff * weight_func * w_tau
            
        return integral * C_q

    def _compute_liouville_entry(self, traj_i, traj_j):
        """
        Оркестратор. Вычисляет интеграл, выбирая метод по названию.
        """
        if self.integration_method == 'quad':
            return self._compute_liouville_entry_quad(traj_i, traj_j)
        elif self.integration_method == 'gaussian':
            return self._compute_liouville_entry_gaussian(traj_i, traj_j)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        
    def fit(self, trajectories=None):
        """
        Строит матричное представление оператора и находит собственные числа.
        
        Если trajectories не переданы, использует те, что сохранены в okhs_transformer.
        okhs_transformer должен быть предварительно обучен (fit).
        """
        if trajectories is None:
            if not hasattr(self.okhs, 'trajectories_'):
                raise ValueError("OKHSTransformer must be fitted first or trajectories provided.")
            trajectories = self.okhs.trajectories_
        
        n_traj = len(trajectories)
        
        # Строим матрицу "A_hat" = [<A mu_j, mu_i>]
        # индексы i, j могут быть несимметричными
        A_hat = np.zeros((n_traj, n_traj))
        
        print(f"Computing Liouville operator matrix ({n_traj}x{n_traj})...")
        for i in range(n_traj):
            for j in range(n_traj):
                A_hat[i, j] = self._compute_liouville_entry(
                    trajectories[i], trajectories[j]
                )
                
        # Получаем матрицу Грама из трансформера
        G = self.okhs.gram_matrix_
        
        # Решаем обобщенную задачу на собственные значения или умножаем на G^-1
        print("Solving eigenvalue problem...")
        try:
            # eig(A, B) решает A*v = lambda*B*v
            eigenvalues, eigenvectors = eig(A_hat, G)
        except Exception as e:
            print(f"Generalized eig failed ({e}), falling back to pinv...")
            # Fallback: явное обращение
            L_mat = np.linalg.pinv(G) @ A_hat
            eigenvalues, eigenvectors = np.linalg.eig(L_mat)
            
        # Сортировка по модулю собственных значений (наиболее значимые - с большим модулем)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        self.liouville_matrix_ = A_hat 
        
        return self

    def get_eigenfunctions(self):
        """
        Возвращает собственные значения и собственные векторы.
        Собственные векторы здесь - это коэффициенты разложения по базису occupation kernels.
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("Operator not fitted yet.")
        return self.eigenvalues_, self.eigenvectors_


class FractionalDMD(BaseEstimator, RegressorMixin):
    """
    Dynamic Mode Decomposition для дробных систем с использованием OKHS.
    
    Вычисляет моды путем проекции тождественного наблюдателя g_id(x) = x
    на собственные функции дробного оператора Лиувилля.
    """

    def __init__(self, liouville_operator):
        self.liouville_operator = liouville_operator
        self.modes_ = None
        self.eigenvalues_ = None
        # Для удобства сохраняем параметры из okhs
        self.q = liouville_operator.okhs.q
        self.C_q = liouville_operator.okhs.C_q

    def _mittag_leffler(self, z, alpha, beta=1):
        """Аппроксимация функции Миттаг-Леффлера."""
        res = np.zeros_like(z, dtype=np.complex128)
        k = 0
        # Ограничиваем число членов ряда для производительности
        while k < 100:
            term = (z ** k) / gamma(alpha * k + beta)
            res += term
            if np.all(np.abs(term) < 1e-15):
                break
            k += 1
        return res

    def _integrate_trajectory_vector(self, trajectory):
        """
        Вычисляет <g_id, phi_gamma> = C_q * ∫ (T-t)^(q-1) * gamma(t) dt.
        Это вектор-столбец размерности n_features.
        """
        T = 1.0  # Нормированное время
        n_features = trajectory.shape[1]
        result = np.zeros(n_features)
        
        # Используем метод интегрирования из оператора или трансформера
        # Для простоты реализуем здесь гауссову квадратуру, как наиболее эффективную
        from numpy.polynomial.legendre import leggauss
        n_points = self.liouville_operator.n_gaussian_points
        nodes, weights = leggauss(n_points)
        
        # Масштабируем узлы на [0, T]
        nodes_t = T * (nodes + 1) / 2
        weights_t = weights * T / 2
        
        evaluate = self.liouville_operator.okhs._evaluate_trajectory_at_time
        
        for t, w in zip(nodes_t, weights_t):
            val = evaluate(trajectory, t)  # (n_features,)
            weight_term = (T - t) ** (self.q - 1)
            result += val * weight_term * w
            
        return result * self.C_q

    def _evaluate_basis_kernel_at_point(self, trajectory_j, x0):
        """
        Вычисляет значение базисной функции (occupation kernel) phi_j в точке x0.
        phi_j(x0) = <K(., x0), phi_j> = C_q * ∫ (T-t)^(q-1) * K(x0, gamma_j(t)) dt
        """
        T = 1.0
        kernel_func = self.liouville_operator.okhs.kernel._compute_single_kernel
        evaluate = self.liouville_operator.okhs._evaluate_trajectory_at_time
        
        # Гауссова квадратура
        from numpy.polynomial.legendre import leggauss
        n_points = self.liouville_operator.n_gaussian_points
        nodes, weights = leggauss(n_points)
        nodes_t = T * (nodes + 1) / 2
        weights_t = weights * T / 2
        
        integral = 0.0
        for t, w in zip(nodes_t, weights_t):
            xt = evaluate(trajectory_j, t)
            k_val = kernel_func(x0, xt)
            weight_term = (T - t) ** (self.q - 1)
            integral += k_val * weight_term * w
            
        return integral * self.C_q

    def fit(self, trajectories=None):
        """
        Обучение модели.
        1. Получает собственные функции оператора Лиувилля (коэффициенты V).
        2. Вычисляет проекции g_id на базис occupation kernels (матрица Y).
        3. Решает систему для нахождения мод Xi: (V^T G V) * Xi = V^T Y.
        """
        # Если траектории не переданы, берем из обученного оператора
        if trajectories is None:
            trajectories = self.liouville_operator.okhs.trajectories_
            
        # Получаем собственные значения и векторы (коэффициенты разложения по phi)
        # eigenvecs - это матрица V, где столбцы - это v_j
        eigenvals, eigenvecs = self.liouville_operator.get_eigenfunctions()
        self.eigenvalues_ = eigenvals
        
        n_traj = len(trajectories)
        n_modes = len(eigenvals)
        n_features = trajectories[0].shape[1]
        
        # 1. Вычисляем матрицу Y проекций g_id на occupation kernels
        # Y[k, :] = <g_id, phi_k>
        print("Computing projections of identity observable...")
        Y = np.zeros((n_traj, n_features))
        for k in range(n_traj):
            Y[k, :] = self._integrate_trajectory_vector(trajectories[k])
            
        # 2. Вычисляем проекции g_id на собственные функции Psi
        # B[j, :] = <g_id, Psi_j> = sum_k v_jk * <g_id, phi_k> = V^T * Y
        # V (n_traj, n_modes), Y (n_traj, n_features) -> B (n_modes, n_features)
        B = eigenvecs.T @ Y
        
        # 3. Вычисляем матрицу Грама собственных функций W = <Psi_i, Psi_j>
        # W = V^T * G * V
        G = self.liouville_operator.okhs.gram_matrix_
        W = eigenvecs.T @ G @ eigenvecs
        
        # 4. Находим моды Xi, решая систему W * Xi = B
        # Xi - матрица (n_modes, n_features)
        print("Solving for Fractional DMD modes...")
        try:
            self.modes_ = np.linalg.solve(W, B)
        except np.linalg.LinAlgError:
            print("Warning: Eigenfunction Gram matrix singular, using pinv.")
            self.modes_ = np.linalg.pinv(W) @ B
            
        return self

    def predict(self, x0, t_span):
        """
        Предсказание траектории для начального условия x0.
        x(t) = Sum_j [ Xi_j * Psi_j(x0) * E_q(lambda_j * t^q) ]
        """
        if self.modes_ is None:
            raise RuntimeError("Model not fitted.")
            
        x0 = np.array(x0)
        t_span = np.array(t_span)
        n_steps = len(t_span)
        n_features = len(x0)
        n_modes = len(self.eigenvalues_)
        
        # 1. Вычисляем значения собственных функций в точке x0
        # Psi_j(x0) = sum_k V_{kj} * phi_k(x0)
        trajectories = self.liouville_operator.okhs.trajectories_
        eigenvecs = self.liouville_operator.eigenvectors_
        
        # Сначала вычислим значения всех базисных ядер в точке x0
        phi_values = np.zeros(len(trajectories))
        for k, traj in enumerate(trajectories):
            phi_values[k] = self._evaluate_basis_kernel_at_point(traj, x0)
            
        # Теперь значения собственных функций: (n_modes,)
        psi_values_x0 = eigenvecs.T @ phi_values
        
        # 2. Суммируем вклады мод
        x_pred = np.zeros((n_steps, n_features), dtype=np.complex128)
        t_q = t_span.astype(np.complex128) ** self.q
        
        for j in range(n_modes):
            lam = self.eigenvalues_[j]
            mode_vec = self.modes_[j, :]  # (n_features,)
            val_at_x0 = psi_values_x0[j]   # scalar
            
            # Временная эволюция
            ml_val = self._mittag_leffler(lam * t_q, self.q) # (n_steps,)
            
            # Вклад j-й моды: (n_steps, n_features)
            # contribution(t) = (Xi_j * Psi_j(x0)) * ML(t)
            # mode_vec * val_at_x0 - это "амплитуда" моды, скорректированная на нач. условие
            term = np.outer(ml_val, mode_vec * val_at_x0)
            x_pred += term
            
        return np.real(x_pred)
