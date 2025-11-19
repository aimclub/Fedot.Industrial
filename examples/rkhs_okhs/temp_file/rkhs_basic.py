import numpy as np

from fedot_ind.core.operation.decomposition.matrix_decomposition.dmd.dmd import FractionalDMD
from fedot_ind.core.operation.transformation.representation.kernel.kernels import OccupationKernel, MultiKernelEnsemble


def create_trajectory():
    # Создаем тестовые траектории
    trajectories = [
        np.sin(np.linspace(0, 2 * np.pi, 20)) + 0.1 * np.random.normal(size=20),
        np.cos(np.linspace(0, 2 * np.pi, 20)) + 0.1 * np.random.normal(size=20),
        np.sin(np.linspace(0, 2 * np.pi, 20) + 0.5) + 0.1 * np.random.normal(size=20)
    ]
    return trajectories


def create_lagged_ts():
    # Создаем временной ряд с памятью
    t = np.linspace(0, 10, 100)
    data = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * np.random.normal(size=100)

    # Разбиваем на траектории
    trajectories = [data[i:i + 20] for i in range(0, 80, 5)]
    return trajectories


def demo_occupation_kernel():
    """Демонстрация Occupation Kernel"""
    print("=== Демонстрация Occupation Kernel ===")

    trajectories = create_trajectory()

    # Occupation Kernel
    kernel = OccupationKernel(q=0.7)
    gram_matrix = kernel.compute_gram_matrix(trajectories)

    print("Матрица Грама:")
    print(gram_matrix)

    # Multi-Kernel Ensemble
    # ensemble = MultiKernelEnsemble()
    # combined_gram = ensemble.compute_gram_matrix(trajectories)

    # print("\nКомбинированная матрица Грама:")
    # print(combined_gram)

    # NotImplementedError for MultiKernelEnsemble compute_gram_matrix

def demo_fractional_forecasting():
    """Демонстрация дробного прогнозирования"""
    print("\n=== Демонстрация Fractional DMD ===")

    trajectories = create_lagged_ts()
    # Fractional DMD
    dmd = FractionalDMD(q=0.8, n_modes=5)
    dmd.fit(trajectories)

    # Прогноз
    future_times = np.arange(20, 30)
    prediction = dmd.predict(trajectories[-1], future_times)

    print(f"Прогноз на {len(future_times)} шагов:")
    print(prediction.flatten())


if __name__ == "__main__":
    demo_occupation_kernel()
    demo_fractional_forecasting()
