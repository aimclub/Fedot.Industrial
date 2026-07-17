import numpy as np

from examples.real_world_examples.benchmark_example.rkhs_okhs.forecasting.example_visuals import (
    build_okhs_visualization_payload,
    function_1,
    function_2,
    generate_sample_trajectory,
)


def test_generate_sample_trajectory_has_expected_shape():
    time_points, trajectory = generate_sample_trajectory()

    assert time_points.shape == (100,)
    assert trajectory.shape == (100, 2)


def test_visual_functionals_match_expected_values():
    point = np.array([1.0, 0.5])

    assert function_1(point) == 1.25
    assert np.isclose(function_2(point), np.sin(1.0) * np.cos(0.5))


def test_visualization_payload_is_side_effect_free():
    time_points, trajectory = generate_sample_trajectory()
    payload = build_okhs_visualization_payload(time_points[:3], trajectory[:3])

    assert set(payload) == {"time_points", "trajectory", "function_values"}
    assert len(payload["time_points"]) == 3
    assert len(payload["function_values"]["x_squared_plus_y_squared"]) == 3
