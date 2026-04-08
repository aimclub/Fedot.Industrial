import numpy as np

from examples.rkhs_okhs.example_visuals import (
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
