import numpy as np
from sklearn.gaussian_process.kernels import RBF


class OccupationKernelFunctional:
    """
    Функционал Occupation Kernel, вычисляющий взвешенную сумму функции вдоль траектории
    """

    def __init__(self, trajectory, time_points=None, kernel=RBF(1.0)):
        """
        Args:
            trajectory: numpy array формы (n_points, n_dimensions) - точки траектории
            time_points: массив временных меток (если None, используется равномерная сетка)
            kernel: ядро для вычисления сходства (по умолчанию RBF)
        """
        self.trajectory = np.array(trajectory)
        self.kernel = kernel

        if time_points is None:
            self.time_points = np.arange(len(trajectory))
        else:
            self.time_points = np.array(time_points)

        # Вычисляем веса для численного интегрирования
        self.weights = self._compute_integration_weights()

    def _compute_integration_weights(self):
        """Вычисляет веса для численного интегрирования методом трапеций"""
        if len(self.time_points) == 1:
            return np.array([1.0])

        weights = np.zeros(len(self.time_points))
        weights[0] = (self.time_points[1] - self.time_points[0]) / 2
        weights[-1] = (self.time_points[-1] - self.time_points[-2]) / 2

        for i in range(1, len(self.time_points) - 1):
            weights[i] = (self.time_points[i + 1] - self.time_points[i - 1]) / 2

        return weights

    def __call__(self, function):
        """
        Вычисляет значение функционала для заданной функции

        Args:
            function: callable, принимающая точку и возвращающая значение

        Returns:
            float: значение функционала (интеграл функции вдоль траектории)
        """
        # Вычисляем значения функции во всех точках траектории
        function_values = np.array([function(point) for point in self.trajectory])

        # Вычисляем взвешенную сумму (численное интегрирование)
        integral = np.sum(function_values * self.weights)

        return integral

    def as_linear_operator(self, test_points):
        """
        Представляет функционал как линейный оператор в RKHS

        Args:
            test_points: точки, в которых определено ядро

        Returns:
            vector: вектор коэффициентов для линейной комбинации ядер
        """
        # Для каждой тестовой точки вычисляем "вклад" траектории
        coefficients = np.zeros(len(test_points))

        for i, test_point in enumerate(test_points):
            # Вычисляем ядро между тестовой точкой и всеми точками траектории
            kernel_values = np.array([self.kernel(test_point, traj_point)
                                      for traj_point in self.trajectory])

            # Взвешенная сумма значений ядра вдоль траектории
            coefficients[i] = np.sum(kernel_values * self.weights)

        return coefficients


class VectorFieldLearner:
    """
    Класс для обучения векторного поля по траекториям с использованием Occupation Kernels
    """

    def __init__(self, kernel=RBF(1.0), regularization=1e-6):
        self.kernel = kernel
        self.regularization = regularization
        self.occupation_functionals = []
        self.velocity_vectors = []
        self.alpha_coefficients = None
        self.reference_points = None

    def add_trajectory(self, trajectory, velocities, time_points=None):
        """
        Добавляет траекторию для обучения

        Args:
            trajectory: массив точек траектории
            velocities: массив скоростей (производных) в каждой точке
            time_points: временные метки
        """
        # Создаем функционал occupation kernel для этой траектории
        ok_func = OccupationKernelFunctional(trajectory, time_points, self.kernel)
        self.occupation_functionals.append(ok_func)

        # Сохраняем среднюю скорость вдоль траектории (целевой вектор)
        avg_velocity = np.mean(velocities, axis=0)
        self.velocity_vectors.append(avg_velocity)

    def fit(self, reference_points):
        """
        Обучает модель векторного поля

        Args:
            reference_points: опорные точки для представления векторного поля
        """
        self.reference_points = reference_points
        n_trajectories = len(self.occupation_functionals)
        n_reference = len(reference_points)
        dim = self.velocity_vectors[0].shape[0]

        # Строим матрицу Грама для функционалов
        gram_matrix = np.zeros((n_trajectories, n_trajectories))

        for i in range(n_trajectories):
            for j in range(n_trajectories):
                # Вычисляем скалярное произведение функционалов
                # через их представление как линейных операторов
                coeffs_i = self.occupation_functionals[i].as_linear_operator(reference_points)
                coeffs_j = self.occupation_functionals[j].as_linear_operator(reference_points)

                # Вычисляем K_{ij} = ⟨O_i, O_j⟩
                kernel_matrix = np.zeros((n_reference, n_reference))
                for k in range(n_reference):
                    for l in range(n_reference):
                        kernel_matrix[k, l] = self.kernel(reference_points[k], reference_points[l])

                gram_matrix[i, j] = coeffs_i @ kernel_matrix @ coeffs_j

        # Решаем задачу для каждой компоненты векторного поля отдельно
        self.alpha_coefficients = np.zeros((n_reference, dim))

        for d in range(dim):
            # Целевой вектор для d-й компоненты
            target = np.array([v[d] for v in self.velocity_vectors])

            # Решаем систему (G + λI)α = y
            self.alpha_coefficients[:, d] = np.linalg.solve(
                gram_matrix + self.regularization * np.eye(n_trajectories),
                target
            )

    def predict(self, query_point):
        """
        Предсказывает векторное поле в заданной точке
        """
        if self.alpha_coefficients is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")

        # Вычисляем вклад всех опорных точек
        result = np.zeros_like(query_point)

        for i, ref_point in enumerate(self.reference_points):
            kernel_value = self.kernel(query_point, ref_point)
            result += self.alpha_coefficients[i] * kernel_value

        return result
