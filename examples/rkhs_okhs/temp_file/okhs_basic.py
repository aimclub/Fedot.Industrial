from sklearn.gaussian_process.kernels import RBF

from examples.rkhs_okhs.temp_file.utils import generate_sample_trajectory, function_2, function_1, vis_okhs_result
from fedot_ind.core.operation.transformation.representation.kernel.occupation import OccupationKernelFunctional

if __name__ == "__main__":
    # Демонстрация работы
    print("=== Демонстрация Occupation Kernel Functional ===")

    # Генерируем данные
    time_points, trajectory = generate_sample_trajectory()

    # Создаем функционал
    ok_functional = OccupationKernelFunctional(
        trajectory=trajectory,
        time_points=time_points,
        kernel=RBF(1.0)
    )

    # Тестируем на разных функциях
    result1 = ok_functional(function_1)
    result2 = ok_functional(function_2)

    print(f"Результат для функции 1 (x² + y²): {result1:.4f}")
    print(f"Результат для функции 2 (sin(x)cos(y)): {result2:.4f}")
    vis_okhs_result(time_points, trajectory)
