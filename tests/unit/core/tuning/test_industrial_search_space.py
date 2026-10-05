from types import SimpleNamespace

from fedot_ind.core.tuning.search_space import (
    build_industrial_pipeline_search_space,
    get_industrial_search_space,
)


def test_search_space_calls_return_independent_values():
    first = get_industrial_search_space()
    second = get_industrial_search_space()

    first["industrial_stat_clf"]["new"] = {"type": "discrete"}

    assert "new" not in second["industrial_stat_clf"]
    assert first["pdl_clf"] is not first["rf"]


def test_custom_search_space_is_merged_without_mutating_input():
    custom = {"industrial_stat_clf": {"custom": {"type": "categorical"}}}
    config = SimpleNamespace(custom_search_space=custom, replace_default_search_space=False)

    result = get_industrial_search_space(config)

    assert result["industrial_stat_clf"]["custom"] == {"type": "categorical"}
    assert custom == {"industrial_stat_clf": {"custom": {"type": "categorical"}}}


def test_builder_uses_an_explicit_complete_space():
    search_space = build_industrial_pipeline_search_space()

    assert "industrial_stat_reg" in search_space.parameters_per_operation
    assert "pdl_reg" in search_space.parameters_per_operation
