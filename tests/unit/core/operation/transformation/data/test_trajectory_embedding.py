import numpy as np

from fedot_ind.core.operation.transformation.data.trajectory_embedding import (
    build_hankel,
    build_page,
    decode_diagonal_average,
    decode_page,
    split_multivariate,
    stack_multivariate,
    truncate_rank,
)


def test_build_hankel_returns_row_major_windows():
    result = build_hankel(np.arange(8, dtype=float), window_size=3)

    assert result.matrix.shape == (6, 3)
    assert np.allclose(result.matrix[0], np.array([0.0, 1.0, 2.0]))
    assert result.diagnostics.kind == 'hankel'


def test_diagonal_average_reconstructs_original_series():
    series = np.arange(8, dtype=float)
    hankel = build_hankel(series, window_size=3).matrix

    reconstructed = decode_diagonal_average(hankel)

    assert np.allclose(reconstructed, series)


def test_page_decode_round_trip_preserves_prefix():
    series = np.arange(12, dtype=float)
    page = build_page(series, window_size=4).matrix

    reconstructed = decode_page(page, original_length=len(series))

    assert np.allclose(reconstructed, series)


def test_multivariate_stacking_and_split_round_trip():
    first = build_page(np.arange(12, dtype=float), window_size=4).matrix
    second = build_page(np.arange(100, 112, dtype=float), window_size=4).matrix

    stacked = stack_multivariate((first, second))
    split = split_multivariate(stacked, channel_count=2)

    assert np.allclose(split[0], first)
    assert np.allclose(split[1], second)


def test_truncate_rank_returns_projected_state_and_basis():
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )

    truncated = truncate_rank(matrix, rank=1)

    assert truncated.selected_rank == 1
    assert truncated.projected_states.shape == (3, 1)
    assert truncated.basis.shape == (3, 1)
    assert truncated.reconstructed_matrix.shape == matrix.shape
