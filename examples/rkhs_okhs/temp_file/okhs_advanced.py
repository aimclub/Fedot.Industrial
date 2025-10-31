import numpy as np
from sklearn.gaussian_process.kernels import RBF

from examples.rkhs_okhs.temp_file.utils import generate_sample_trajectory
from fedot_ind.core.operation.transformation.representation.kernel.occupation import VectorFieldLearner

if __name__ == "__main__":
    print("\n=== Пример обучения векторного поля ===")

    # Генерируем несколько траекторий
    trajectories = []
    velocities = []

    for _ in range(5):
        t, traj = generate_sample_trajectory()

        # Вычисляем численные производные (скорости)
        dt = t[1] - t[0]
        vel = np.gradient(traj, dt, axis=0)

        trajectories.append(traj)
        velocities.append(vel)

    # Создаем и обучаем модель
    learner = VectorFieldLearner(kernel=RBF(0.5), regularization=1e-8)

    for traj, vel in zip(trajectories, velocities):
        learner.add_trajectory(traj, vel)

    # Опорные точки для представления векторного поля
    x_grid, y_grid = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(-1.5, 1.5, 5))
    reference_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    learner.fit(reference_points)

    # Тестируем предсказание
    test_point = np.array([0.5, 0.3])
    predicted_vector = learner.predict(test_point)
    print(f"Предсказанное векторное поле в точке {test_point}: {predicted_vector}")
