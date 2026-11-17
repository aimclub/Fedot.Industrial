"""Public helpers for describing a training series and recommending a model."""

from .diagnostics import RegimeDiagnosticsResult, analyze_regime_diagnostics
from .routing import RegimeRoutingDecision, RegimeRoutingPolicy, recommend_forecasting_model

__all__ = [
    "RegimeDiagnosticsResult", "RegimeRoutingDecision", "RegimeRoutingPolicy",
    "analyze_regime_diagnostics", "recommend_forecasting_model",
]
