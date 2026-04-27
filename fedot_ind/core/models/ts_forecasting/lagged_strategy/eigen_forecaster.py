from fedot_ind.core.models.ts_forecasting.lagged_model.low_rank_lagged_ridge_forecaster import (
    LowRankLaggedRidgeForecasterImplementation,
)


class EigenAR(LowRankLaggedRidgeForecasterImplementation):
    """Compatibility alias for the historical low-rank lagged forecaster entrypoint."""
