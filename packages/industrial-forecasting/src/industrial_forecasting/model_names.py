"""Forecast model identifiers shared by tuning and integration adapters."""

FORECASTING_MODEL_ALIASES: dict[str, str] = {
    'mssa': 'mssa_forecaster',
    'havok': 'havok_forecaster',
}


def canonical_forecasting_model_name(name: str | None) -> str:
    normalized = str(name or '').strip().lower()
    return FORECASTING_MODEL_ALIASES.get(normalized, normalized)


def forecasting_aliases_for(model_name: str) -> tuple[str, ...]:
    canonical = canonical_forecasting_model_name(model_name)
    aliases = [alias for alias, target in FORECASTING_MODEL_ALIASES.items() if target == canonical]
    return tuple(dict.fromkeys([canonical, *sorted(aliases)]))
