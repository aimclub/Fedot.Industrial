"""Compatibility imports for the standalone trajectory primitives."""

from industrial_forecasting.trajectory import (
    RankTruncationResult,
    TrajectoryEmbeddingDiagnostics,
    TrajectoryEmbeddingResult,
    build_hankel,
    build_page,
    decode_diagonal_average,
    decode_page,
    estimate_window,
    normalize_multivariate_series,
    split_multivariate,
    stack_multivariate,
    truncate_rank,
)

__all__ = [
    'RankTruncationResult', 'TrajectoryEmbeddingDiagnostics', 'TrajectoryEmbeddingResult',
    'build_hankel', 'build_page', 'decode_diagonal_average', 'decode_page', 'estimate_window',
    'normalize_multivariate_series', 'split_multivariate', 'stack_multivariate', 'truncate_rank',
]
