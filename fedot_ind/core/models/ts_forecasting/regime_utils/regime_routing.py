"""Compatibility imports for regime routing."""

from industrial_forecasting.regimes.routing import (
    RegimeRoutingDecision, RegimeRoutingPolicy, RoutingAdapterName,
    adapter_name_to_family, recommend_forecasting_model,
)

__all__ = [
    'RegimeRoutingDecision', 'RegimeRoutingPolicy', 'RoutingAdapterName',
    'adapter_name_to_family', 'recommend_forecasting_model',
]
