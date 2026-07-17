from __future__ import annotations

import numpy as np


def build_okhs_visualization_payload(time_points, trajectory) -> dict[str, object]:
    function_1_values = [function_1(point) for point in trajectory]
    function_2_values = [function_2(point) for point in trajectory]
    return {
        "time_points": np.asarray(time_points).tolist(),
        "trajectory": np.asarray(trajectory).tolist(),
        "function_values": {
            "x_squared_plus_y_squared": function_1_values,
            "sin_x_cos_y": function_2_values,
        },
    }


def generate_sample_trajectory():
    time_points = np.linspace(0, 4 * np.pi, 100)
    x_values = np.cos(time_points)
    y_values = np.sin(0.5 * time_points)
    trajectory = np.column_stack([x_values, y_values])
    return time_points, trajectory


def function_1(point):
    return point[0] ** 2 + point[1] ** 2


def function_2(point):
    return np.sin(point[0]) * np.cos(point[1])


__all__ = [
    "build_okhs_visualization_payload",
    "function_1",
    "function_2",
    "generate_sample_trajectory",
]
