_EXPORTS = {
    "build_scenario_context": "examples.real_world_examples.industrial_examples.current_api",
    "build_kernel_learning_reference_model_specs": "examples.real_world_examples.industrial_examples.current_api",
    "build_scenario_forecast_preview": "examples.real_world_examples.industrial_examples.current_api",
    "build_scenario_model_specs": "examples.real_world_examples.industrial_examples.current_api",
    "build_scenario_preview_frame": "examples.real_world_examples.industrial_examples.current_api",
    "list_domain_scenarios": "examples.real_world_examples.industrial_examples.current_api",
    "load_scenario_defaults": "examples.real_world_examples.industrial_examples.current_api",
    "render_scenario_forecast_pack": "examples.real_world_examples.industrial_examples.current_api",
    "render_scenario_preview_pack": "examples.real_world_examples.industrial_examples.current_api",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
