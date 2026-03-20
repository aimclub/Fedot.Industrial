import numpy as np


def vis_okhs_result(time_points, trajectory):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.7)
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=time_points, cmap="viridis", s=20)
    plt.colorbar(label="Time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Траектория временного ряда")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    function_1_values = [function_1(point) for point in trajectory]
    function_2_values = [function_2(point) for point in trajectory]

    plt.plot(time_points, function_1_values, "r-", label="x^2 + y^2", alpha=0.7)
    plt.plot(time_points, function_2_values, "g-", label="sin(x)cos(y)", alpha=0.7)
    plt.fill_between(time_points, function_1_values, alpha=0.2, color="red")
    plt.fill_between(time_points, function_2_values, alpha=0.2, color="green")
    plt.xlabel("Время")
    plt.ylabel("Значение функции")
    plt.title("Функции вдоль траектории\n(площадь под кривой = значение функционала)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_sample_trajectory():
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(0.5 * t)
    trajectory = np.column_stack([x, y])
    return t, trajectory


def function_1(point):
    return point[0] ** 2 + point[1] ** 2


def function_2(point):
    return np.sin(point[0]) * np.cos(point[1])
