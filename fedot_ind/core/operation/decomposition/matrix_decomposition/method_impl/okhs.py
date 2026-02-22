import numpy as np
from sklearn.base import TransformerMixin
from scipy.integrate import quad
from scipy.special import gamma, roots_jacobi
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import eig
from pymittagleffler import mittag_leffler
# from ..core.kernels import OccupationKernel
# from ....transformation.representation.kernel.kernels import OccupationKernel

class OKHSTransformer(TransformerMixin, BaseEstimator):
    """
    Трансформер для OKHS признаков с использованием физического времени
    и квадратур Гаусса-Якоби для учета сингулярностей.
    
    Матрица Грама вычисляется как:
    G_{i,j} = C_q^2 * ∫∫_{[0,T]²} (T - τ)^{q-1} (T - t)^{q-1} K(ξ_j(t), ξ_i(τ)) dt dτ
    
    С использованием квадратур Якоби сингулярный вес (T-t)^{q-1} учитывается
    в весах квадратуры w_k, что дает точное интегрирование для полиномиальных ядер.
    """

    def __init__(self, kernel, q=0.7, n_quad_points=20, dt=1.0):
        """
        Параметры:
        -----------
        kernel : KernelBase
            Ядро (например, RBFKernel). Должно иметь метод _compute_single_kernel(x, y).
        
        q : float (0 < q ≤ 1)
            Порядок дробной производной.
        
        n_quad_points : int
            Количество точек для метода 'jacobi'.
            
        dt : float
            Шаг дискретизации времени траекторий. Используется для вычисления T.
            Если данные не имеют временной метки, dt=1.0 означает, что T = n_steps.
        """
        self.kernel = kernel
        self.q = q
        self.n_quad_points = n_quad_points
        self.dt = dt
        self.C_q = 1.0 / gamma(q)
        
        # Кэш для узлов и весов квадратуры
        self._quad_cache = None

    def _get_trajectory_duration(self, trajectory):
        """
        Возвращает физическую длительность траектории T.
        """
        n_steps = len(trajectory)
        # T - это интервал времени от 0 до конца.
        # Если 5 точек с шагом 0.1, то T = 0.4 (t: 0.0, 0.1, 0.2, 0.3, 0.4)
        return (n_steps - 1) * self.dt

    def _evaluate_trajectory_at_time(self, trajectory, t, T):
        """
        Интерполяция траектории в физический момент времени t ∈ [0, T].
        """
        if T <= 1e-14: # Защита от деления на ноль для вырожденных траекторий
            return trajectory[0]
            
        # Нормализуем время к индексу массива
        # t / T дает долю пути (0..1), умножаем на (N-1) чтобы получить индекс
        n_steps = len(trajectory)
        t_idx = (t / T) * (n_steps - 1)
        
        idx = int(np.floor(t_idx))
        idx = np.clip(idx, 0, n_steps - 2)
        
        alpha = t_idx - idx
        
        # Линейная интерполяция
        value = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
        return value

    def _get_jacobi_rule(self):
        """
        Возвращает узлы (x) и веса (w) квадратуры Гаусса-Якоби для интеграла:
        ∫_{-1}^{1} (1-x)^alpha (1+x)^beta f(x) dx.
        
        Нам нужно интегрировать (T - t)^(q-1).
        При замене переменной t = T(x+1)/2, член (T-t) пропорционален (1-x).
        Поэтому alpha = q - 1, beta = 0.
        """
        if self._quad_cache is None:
            # alpha=q-1, beta=0. 
            # Функция roots_jacobi возвращает узлы для веса (1-x)^alpha (1+x)^beta
            self._quad_cache = roots_jacobi(self.n_quad_points, self.q - 1, 0)
        return self._quad_cache

    def _compute_gram_entry_jacobi(self, trajectory_i, trajectory_j):
        """
        Вычисление элемента через квадратуры Гаусса-Якоби.
        
        Интеграл вида: I = ∫_0^T (T-t)^(q-1) f(t) dt.
        Замена: t = T(x+1)/2, dt = (T/2)dx.
        (T - t) = T - T(x+1)/2 = T(1 - (x+1)/2) = T(1-x)/2.
        
        Тогда (T-t)^(q-1) = (T/2)^(q-1) * (1-x)^(q-1).
        
        I = ∫_{-1}^1 (T/2)^(q-1) (1-x)^(q-1) f(T(x+1)/2) * (T/2) dx
          = (T/2)^q ∫_{-1}^1 (1-x)^(q-1) f(...) dx
          
        Интеграл квадратуры Якоби: Sum w_k * f(x_k). 
        Вес (1-x)^(q-1) уже "сидит" в w_k.
        Остается только множитель (T/2)^q.
        """
        T_i = self._get_trajectory_duration(trajectory_i)
        T_j = self._get_trajectory_duration(trajectory_j)
        T = min(T_i, T_j)
        
        nodes, weights = self._get_jacobi_rule()
        
        # Масштабирование узлов с [-1, 1] на [0, T]
        # t_k = T * (x_k + 1) / 2
        t_nodes = T * (nodes + 1) / 2
        
        vals_i = np.array([self._evaluate_trajectory_at_time(trajectory_i, t, T) for t in t_nodes])
        vals_j = np.array([self._evaluate_trajectory_at_time(trajectory_j, t, T) for t in t_nodes])
        
        # Якобиан преобразования для одного интеграла: (T/2)^q
        jacobian_pow_q = (T / 2.0) ** self.q
        
        gram_entry = 0.0
        
        # Двойная сумма
        # G ~ C_q^2 * J^2 * sum_k sum_m w_k w_m K(xi_j(t_m), xi_i(tau_k))
        # Здесь J^2 = (T/2)^(2q), так как интеграл двойной
        
        for k in range(self.n_quad_points):
            w_tau = weights[k]
            xi_tau = vals_i[k]
            
            for m in range(self.n_quad_points):
                w_t = weights[m]
                xi_t = vals_j[m]
                
                # Ядро (без сингулярных весов - они учтены в w_tau, w_t)
                K_val = self.kernel._compute_single_kernel(xi_t, xi_tau)
                
                gram_entry += w_tau * w_t * K_val
                
        # Полный множитель: C_q^2 * ((T/2)^q)^2
        scale_factor = (self.C_q * jacobian_pow_q) ** 2
        return gram_entry * scale_factor

    def _compute_gram_matrix(self, trajectories):
        n = len(trajectories)
        gram_matrix = np.zeros((n, n))
        
        # Используем только метод Якоби
        compute = self._compute_gram_entry_jacobi
        
        for i in range(n):
            for j in range(i, n):
                val = compute(trajectories[i], trajectories[j])
                gram_matrix[i, j] = val
                gram_matrix[j, i] = val
                
        return gram_matrix

    def fit(self, train_trajectories, y=None):
        """
        Обучение трансформера: вычисление матрицы Грама.
        
        Parameters
        ----------
        train_trajectories : list of array-like
            Список обучающих траекторий
        """
        self.train_trajectories_ = train_trajectories
        self.gram_matrix_ = self._compute_gram_matrix(train_trajectories)
        
        cond_number = np.linalg.cond(self.gram_matrix_)
        
        # Регуляризация
        if cond_number > 1e10:
            regularization = 1e-8 * np.eye(len(train_trajectories))
            self.gram_matrix_ += regularization
            
        return self

    def transform(self, test_trajectories):
        """
        Вычисляет координаты новых (тестовых) траекторий в базисе обучающих.
        
        Parameters
        ----------
        test_trajectories : list of array-like
            Список тестовых траекторий
        """
        n_train = len(self.train_trajectories_)
        n_test = len(test_trajectories)
        
        K_test_train = np.zeros((n_test, n_train))
        compute = self._compute_gram_entry_jacobi

        for i in range(n_test):
            for j in range(n_train):
                # Вычисляем ядро между тестовой траекторией i и обучающей траекторией j
                K_test_train[i, j] = compute(test_trajectories[i], self.train_trajectories_[j])
                
        # c = (G^-1 K^T)^T = K G^-1
        # G c^T = K^T -> решаем для каждой строки
        try:
            c = np.linalg.solve(self.gram_matrix_, K_test_train.T).T
        except np.linalg.LinAlgError:
            c = K_test_train @ np.linalg.pinv(self.gram_matrix_)
            
        return c
    

class FractionalLiouvilleOperator(BaseEstimator):
    """
    Оператор Лиувилля дробного порядка с использованием квадратур Якоби.
    
    Реализует конечномерное представление оператора P A_{f,q} P.
    Элементы матрицы вычисляются через однократный интеграл с сингулярным весом:
    
    A_{ij} = <A* mu_i, mu_j> 
           = C_q * ∫_0^T (T-τ)^{q-1} [K(ξ_j(τ), ξ_i(T)) - K(ξ_j(τ), ξ_i(0))] dτ
           
    Интеграл вычисляется точно с помощью квадратур Гаусса-Якоби для веса (1-x)^{q-1}.
    """

    def __init__(self, okhs_transformer, n_quad_points=20):
        """
        Parameters
        ----------
        okhs_transformer : OKHSTransformer
            Обученный экземпляр OKHSTransformer. Должен иметь атрибуты q, C_q, dt.
            
        n_quad_points : int
            Число точек квадратуры Якоби для вычисления элементов оператора.
        """
        self.okhs = okhs_transformer
        self.n_quad_points = n_quad_points
        
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.liouville_matrix_ = None
        
        # Кэш для квадратур
        self._quad_cache = None

    def _get_jacobi_rule(self):
        """Возвращает узлы и веса для веса (1-x)^{q-1}."""
        if self._quad_cache is None:
            # alpha = q - 1, beta = 0
            # weight function: (1-x)^alpha * (1+x)^0
            q = self.okhs.q
            self._quad_cache = roots_jacobi(self.n_quad_points, q - 1, 0)
        return self._quad_cache

    def _compute_liouville_entry(self, traj_i, traj_j):
        """
        Вычисляет элемент матрицы A_{ij}.
        """

        T_i = self.okhs._get_trajectory_duration(traj_i)
        T_j = self.okhs._get_trajectory_duration(traj_j)
        T = min(T_i, T_j)
        
        if T <= 1e-14:
            return 0.0

        nodes, weights = self._get_jacobi_rule()
        
        # Масштабируем узлы квадратуры с [-1, 1] на [0, T]
        # tau = T * (x + 1) / 2
        tau_nodes = T * (nodes + 1) / 2
        

        # Используем интерполяцию из трансформера для получения точных граничных значений
        xi_i_T = self.okhs._evaluate_trajectory_at_time(traj_i, T, T_i)
        xi_i_0 = self.okhs._evaluate_trajectory_at_time(traj_i, 0.0, T_i)
        
        # Предвычисляем значения j-й траектории (mu_j) во всех точках интегрирования
        xi_j_vals = [self.okhs._evaluate_trajectory_at_time(traj_j, tau, T_j) for tau in tau_nodes]
        
        kernel_func = self.okhs.kernel._compute_single_kernel
        
        integral_sum = 0.0
        
        for k in range(self.n_quad_points):
            w_k = weights[k]
            xi_j_tau = xi_j_vals[k]
            
            # Разность ядер: K(ξ_j(τ), ξ_i(T)) - K(ξ_j(τ), ξ_i(0))
            term_T = kernel_func(xi_j_tau, xi_i_T)
            term_0 = kernel_func(xi_j_tau, xi_i_0)
            
            integral_sum += w_k * (term_T - term_0)
            
        # Якобиан преобразования: dt = (T/2) dx
        # И множитель (T/2)^(q-1) от весовой функции
        # Итоговый множитель: (T/2)^q
        jacobian_factor = (T / 2.0) ** self.okhs.q
        
        result = self.okhs.C_q * jacobian_factor * integral_sum
        return result

    def fit(self, trajectories=None):
        """
        Строит матрицу оператора и решает обобщенную задачу на собственные значения.
        """
        if trajectories is None:
            if not hasattr(self.okhs, 'train_trajectories_'):
                raise ValueError("OKHSTransformer must be fitted first.")
            trajectories = self.okhs.train_trajectories_
        
        n_traj = len(trajectories)
        self.liouville_matrix_ = np.zeros((n_traj, n_traj))
        
        print(f"Computing Liouville matrix ({n_traj}x{n_traj}) using Jacobi quadratures...")
        
        for i in range(n_traj):
            for j in range(n_traj):
                # Оператор не симметричен, вычисляем все элементы
                self.liouville_matrix_[i, j] = self._compute_liouville_entry(
                    trajectories[i], trajectories[j]
                )
                
        # Получаем матрицу Грама G
        G = self.okhs.gram_matrix_
        
        print("Solving generalized eigenvalue problem A v = lambda G v...")
        try:
            # Решаем A * v = lambda * G * v
            eigenvalues, eigenvectors = eig(self.liouville_matrix_, G)
        except Exception as e:
            print(f"Generalized eig failed ({e}), using pseudo-inverse fallback.")
            L_mat = np.linalg.pinv(G) @ self.liouville_matrix_
            eigenvalues, eigenvectors = np.linalg.eig(L_mat)

        # Нормируем собственные векторы по норме, определенной через G: v^* G v = 1
        for i in range(eigenvectors.shape[1]):
            v = eigenvectors[:, i]
            norm = np.sqrt(np.abs(v.conj().T @ G @ v))
            eigenvectors[:, i] = v / (norm + 1e-12) 

        # Сортировка по модулю собственных значений (от больших к меньшим)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        
        return self

    def get_eigenfunctions(self):
        if self.eigenvalues_ is None:
            raise RuntimeError("Operator not fitted.")
        return self.eigenvalues_, self.eigenvectors_


class FractionalDMD(BaseEstimator, RegressorMixin):
    def __init__(self, liouville_operator, n_quad_points=20, regularization=1e-8):
        self.liouville_operator = liouville_operator
        self.okhs = liouville_operator.okhs
        self.n_quad_points = n_quad_points
        self.regularization = regularization
        
        self.modes_ = None
        self._quad_cache = None


    def _get_jacobi_rule(self):
        if self._quad_cache is None:
            self._quad_cache = roots_jacobi(self.n_quad_points, self.okhs.q - 1, 0)
        return self._quad_cache
        
    def _integrate_observable_projection(self, trajectory, observable_func):
        """
        Вычисляет <g_id, Phi_k>_OKHS.
        Это равно (T g_id)(trajectory) = C_q * int (T-t)^(q-1) g_id(traj(t)) dt.
        """
        T = self.okhs._get_trajectory_duration(trajectory)
        if T <= 1e-14: 
            return np.zeros_like(observable_func(0.0))

        nodes, weights = self._get_jacobi_rule()
        t_nodes = T * (nodes + 1) / 2
        jacobian_factor = (T / 2.0) ** self.okhs.q
        
        integral_sum = 0.0
        for k in range(self.n_quad_points):
            val = observable_func(t_nodes[k])
            integral_sum += weights[k] * val
            
        return self.okhs.C_q * jacobian_factor * integral_sum


    def compute_identity_projections(self, trajectories):
        """
        Вычисляет матрицу Y (проекции g_id на occupation kernels).
        Y_ki = < (g_id)_i, Phi_k >
        """
        n_traj = len(trajectories)
        n_features = trajectories[0].shape[1]
        Y = np.zeros((n_traj, n_features))
        
        for k in range(n_traj):
            traj = trajectories[k]
            T_traj = self.okhs._get_trajectory_duration(traj)
            
            obs_func = lambda t: self.okhs._evaluate_trajectory_at_time(traj, t, T_traj)
            Y[k, :] = self._integrate_observable_projection(traj, obs_func)
            
        return Y


    def compute_eigenfunction_projections(self, Y, V):
        """
        Переход к базису собственных функций.
        B = V^* Y
        """
        return V.conj().T @ Y


    def solve_modes(self, W, B):
        """
        Решение системы W * Xi = B.
        """
        W_reg = W + self.regularization * np.eye(W.shape[0])
        
        try:
            Xi = np.linalg.solve(W_reg, B)
        except np.linalg.LinAlgError:
            Xi = np.linalg.pinv(W_reg) @ B
        return Xi


    def fit(self, trajectories=None):
        if trajectories is None:
            trajectories = self.okhs.train_trajectories_
            
        if self.liouville_operator.eigenvectors_ is None:
             raise ValueError("Liouville operator must be fitted.")
             
        V = self.liouville_operator.eigenvectors_
        G = self.okhs.gram_matrix_
        
        # Compute Y
        Y = self.compute_identity_projections(trajectories)
        
        # Compute B = V^* Y
        B = self.compute_eigenfunction_projections(Y, V)
        
        # Compute W = V^* G V (Gram matrix in eigenbasis)
        W = V.conj().T @ G @ V
        
        # Solve for modes Xi
        self.modes_ = self.solve_modes(W, B)
        
        return self


    def fit_initial_coefficients(self, initial_trajectory):
        """
        Определяет коэффициенты c_j из решения системы уравнений:
        
        x(t_k) = sum_j c_j * Xi_j * E_q(lambda_j * t_k^q)
        
        Используются все точки initial_trajectory. Система решается в смысле 
        наименьших квадратов.
        
        Parameters
        ----------
        initial_trajectory : array-like, shape (K, n_features)
            Начальный сегмент траектории (от t=0).
            
        Returns
        -------
        c : array, shape (n_modes,), dtype=complex
            Коэффициенты разложения.
        """
        initial_trajectory = np.asarray(initial_trajectory)
        K, n_features = initial_trajectory.shape
        
        eig = np.asarray(self.liouville_operator.eigenvalues_)
        Xi = np.asarray(self.modes_)  # (n_modes, n_features)
        n_modes = len(eig)
        
        # Проверка размерности
        if K * n_features < n_modes:
            raise ValueError(
                f"Недостаточно данных для определения {n_modes} коэффициентов: "
                f"K={K} точек × d={n_features} признаков = {K * n_features} уравнений < {n_modes}."
            )
        
        # Временная сетка: t_k = k * dt, k=0..K-1
        t_grid = np.arange(K) * self.okhs.dt
        
        # Строим систему A @ c ≈ b
        # A: (K*n_features, n_modes), b: (K*n_features,)
        A = np.zeros((K * n_features, n_modes), dtype=np.complex128)
        b = initial_trajectory.reshape(K * n_features).astype(np.complex128)
        
        for k, t in enumerate(t_grid):
            ml = mittag_leffler(eig * (t ** self.okhs.q), self.okhs.q, 1)  # (n_modes,)
            
            # блок строк [k*d : (k+1)*d, :]
            # Строка для признака d=0..n_features-1: A[k*d+d, j] = ml[j] * Xi[j, d]
            A[k * n_features:(k + 1) * n_features, :] = (ml[:, None] * Xi).T
        

        # Решаем систему в смысле наименьших квадратов с регуляризацией
        n_modes = A.shape[1]
        alpha = 1e-5

        # Вычисляем A^H * A + alpha * I
        # .conj().T — это эрмитово сопряжение
        A_stack = A.conj().T @ A
        reg_matrix = A_stack + alpha * np.eye(n_modes)

        b_stack = A.conj().T @ b
        try:
            c = np.linalg.solve(reg_matrix, b_stack)
        except np.linalg.LinAlgError:
            c = np.linalg.pinv(reg_matrix) @ b_stack

        self.initial_coefficients_ = c
                
        return c


    def predict(self, initial_trajectory, t_span):
        """
        Предсказание траектории на основе начального сегмента.
        
        Parameters
        ----------
        initial_trajectory : array-like, shape (K, n_features)
            Начальный сегмент траектории (используется для определения c_j).
            
        t_span : array-like, shape (n_predict,)
            Временные точки для предсказания (физическое время от t=0).
            
        Returns
        -------
        x_pred : array, shape (n_predict, n_features)
            Предсказанная траектория.
        """
        initial_trajectory = np.asarray(initial_trajectory)
        t_span = np.asarray(t_span)
        
        eig = np.asarray(self.liouville_operator.eigenvalues_)
        Xi = np.asarray(self.modes_)  # (n_modes, n_features)
        
        c = self.fit_initial_coefficients(initial_trajectory)
        
        # Вычисляем эволюцию Mittag-Leffler для всех t и всех λ
        t_q = (t_span.astype(np.complex128) ** self.okhs.q)[:, None]  # (n_pred, 1)
        lam = eig[None, :]  # (1, n_modes)
        ML = mittag_leffler(lam * t_q, self.okhs.q, 1)  # (n_pred, n_modes)
        
        # x(t) = sum_j c_j * E_q(λ_j t^q) * Xi_j
        # В матричной форме: ML @ diag(c) @ Xi = ML @ (c[:, None] * Xi)

        X = c[:, None] * Xi  # (n_modes, n_features)
        x_pred = ML @ X      # (n_pred, n_features)
        
        return np.real(x_pred)
    

    def plot_predict(self, initial_trajectory, t_span):
        """
        Предсказание траектории и визуализация компонентов модели.
        Отображаются четыре графика:
        1. Действительная часть Mittag-Leffler функций E_q(λ t^q) для наиболее значимых мод.
        2. Амплитуды коэффициентов |c_j| (столбцы).
        3. Амплитуды первых компонент мод Лиувилля |Xi[j,0]|.
        4. Собственные значения λ_j на комплексной плоскости.
        Все элементы, соответствующие одной моде j, имеют одинаковый цвет.

        Parameters
        ----------
        initial_trajectory : array-like, shape (K, n_features)
            Начальный сегмент траектории (используется для определения c_j).
        t_span : array-like, shape (n_predict,)
            Временные точки для предсказания (физическое время от t=0).

        Returns
        -------
        x_pred : array, shape (n_predict, n_features)
            Предсказанная траектория (действительная часть).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.cm import tab10, tab20

        # ----------------------------------------------------------------------
        # 1. Вычисления (без изменений, как в оригинале)
        # ----------------------------------------------------------------------
        initial_trajectory = np.asarray(initial_trajectory)
        t_span = np.asarray(t_span)
        n_pred = len(t_span)
        n_features = initial_trajectory.shape[1]

        eig = np.asarray(self.liouville_operator.eigenvalues_)
        Xi = np.asarray(self.modes_)          # (n_modes, n_features)
        n_modes = len(eig)

        # Коэффициенты c_j из начального сегмента
        c = self.fit_initial_coefficients(initial_trajectory)

        # Mittag-Leffler для всех t и всех λ
        t_q = (t_span.astype(np.complex128) ** self.okhs.q)[:, None]   # (n_pred, 1)
        lam = eig[None, :]                                               # (1, n_modes)
        ML = mittag_leffler(lam * t_q, self.okhs.q, 1)                   # (n_pred, n_modes)

        # Предсказание (действительная часть)
        X = c[:, None] * Xi                     # (n_modes, n_features)
        x_pred = ML @ X                         # (n_pred, n_features)
        x_pred = np.real(x_pred)

        # ----------------------------------------------------------------------
        # 2. Подготовка данных для визуализации
        # ----------------------------------------------------------------------
        # Выбираем топ‑мод по |c_j| (не более 15 для читаемости)
        abs_c = np.abs(c)
        top_k = min(15, n_modes)
        top_idx = np.argsort(abs_c)[::-1][:top_k]

        # Для остальных мод (на графике собственных значений) используем серый цвет
        other_idx = np.setdiff1d(np.arange(n_modes), top_idx)

        # Цветовая карта для топ‑мод (tab10 + tab20 при необходимости)
        if top_k <= 10:
            colors = [tab10(i) for i in range(top_k)]
        else:
            # tab20 содержит 20 цветов, чередующихся светлый/тёмный
            colors = [tab20(i) for i in range(top_k)]

        # ----------------------------------------------------------------------
        # 3. Построение фигуры 2x2
        # ----------------------------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ компонентов модели fDMD (OKHS)', fontsize=16, fontweight='bold')

        # --- 3.1 График Mittag-Leffler (действительная часть) ---
        ax_ml = axs[0, 0]
        for i, j in enumerate(top_idx):
            ml_real = np.real(ML[:, j])
            ax_ml.plot(t_span, ml_real, color=colors[i], linewidth=1.5,
                    label=f'j={j}  λ={eig[j]:.3f}')
        ax_ml.set_xlabel('Время t')
        ax_ml.set_ylabel('Re(E_q(λ t^q))')
        ax_ml.set_title('Mittag-Leffler функции (топ-{})'.format(top_k))
        ax_ml.legend(loc='best', fontsize=8, ncol=2)
        ax_ml.grid(True, alpha=0.3)

        # --- 3.2 Амплитуды коэффициентов |c_j| ---
        ax_c = axs[0, 1]
        x_pos = np.arange(top_k)
        bars_c = ax_c.bar(x_pos, abs_c[top_idx], color=colors, edgecolor='black', linewidth=0.5)
        ax_c.set_xlabel('Индекс моды (отсортированы по |c|)')
        ax_c.set_ylabel('|c_j|')
        ax_c.set_title('Коэффициенты разложения')
        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels([str(j) for j in top_idx], rotation=45)
        # Подписи значений на столбцах (если не слишком мелко)
        for bar, val in zip(bars_c, abs_c[top_idx]):
            ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        # --- 3.3 Первые компоненты мод |Xi[j,0]| ---
        ax_xi = axs[1, 0]
        abs_xi0 = np.abs(Xi[top_idx, 0])   # модуль первой компоненты
        bars_xi = ax_xi.bar(x_pos, abs_xi0, color=colors, edgecolor='black', linewidth=0.5)
        ax_xi.set_xlabel('Индекс моды (отсортированы по |c|)')
        ax_xi.set_ylabel('|Xi[j,0]|')
        ax_xi.set_title('Первая компонента мод Лиувилля')
        ax_xi.set_xticks(x_pos)
        ax_xi.set_xticklabels([str(j) for j in top_idx], rotation=45)
        for bar, val in zip(bars_xi, abs_xi0):
            ax_xi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        # --- 3.4 Собственные значения на комплексной плоскости ---
        ax_eig = axs[1, 1]
        # Все остальные моды серым
        if len(other_idx) > 0:
            ax_eig.scatter(np.real(eig[other_idx]), np.imag(eig[other_idx]),
                        c='gray', alpha=0.5, s=20, label='прочие моды')
        # Топ‑моды цветными точками
        for i, j in enumerate(top_idx):
            ax_eig.scatter(np.real(eig[j]), np.imag(eig[j]),
                        color=colors[i], s=80, edgecolor='black', linewidth=0.5,
                        label=f'j={j}' if i == 0 else "")  # легенда только один раз
        ax_eig.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_eig.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_eig.set_xlabel('Re(λ)')
        ax_eig.set_ylabel('Im(λ)')
        ax_eig.set_title('Собственные значения λ_j')
        ax_eig.grid(True, alpha=0.3)
        # Добавим общую легенду для цветов (если нужно)
        # Можно создать отдельную легенду для топ‑мод, но проще полагаться на цветовую кодировку
        # Если мод много, лучше сделать выноску с индексами рядом с точками
        for i, j in enumerate(top_idx):
            ax_eig.annotate(str(j), (np.real(eig[j]), np.imag(eig[j])),
                            textcoords="offset points", xytext=(5,5), fontsize=6)

        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------------------------
        # 4. Краткая диагностика в консоли
        # ----------------------------------------------------------------------
        print("\n=== Диагностика ===")
        print(f"Всего мод: {n_modes}, показано топ-{top_k} по |c|")
        print(f"Диапазон Re(λ): [{np.min(np.real(eig)):.3f}, {np.max(np.real(eig)):.3f}]")
        print(f"Диапазон Im(λ): [{np.min(np.imag(eig)):.3f}, {np.max(np.imag(eig)):.3f}]")
        # Поиск потенциально взрывных мод (Re(λ) > 0.1)
        unstable = np.where(np.real(eig) > 0.1)[0]
        if len(unstable) > 0:
            print("⚠️  Моды с положительной действительной частью (возможный рост):")
            for j in unstable[:5]:
                print(f"   j={j}, λ={eig[j]:.3f}, |c|={abs_c[j]:.3f}")
        else:
            print("✅ Моды с положительной Re(λ) отсутствуют.")

        return x_pred