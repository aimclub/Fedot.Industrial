from __future__ import annotations

from fedot_ind.api.flow import (
    InitialAssumptionSource,
    normalize_initial_assumption,
    normalize_initial_assumption_from_configs,
    resolve_initial_assumption_plan,
)


class FakePipeline:
    pass


class FakePipelineBuilder:
    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls

    def build(self):
        self.calls.append(self.name)
        return FakePipeline()


def test_normalize_initial_assumption_preserves_none():
    assert normalize_initial_assumption(None) is None


def test_normalize_initial_assumption_does_not_rebuild_ready_pipeline():
    pipeline = FakePipeline()

    assert normalize_initial_assumption(pipeline) is pipeline


def test_normalize_initial_assumption_builds_pipeline_builder_lazily_when_called():
    calls = []
    builder = FakePipelineBuilder("builder", calls)

    assert calls == []
    normalized = normalize_initial_assumption(builder)

    assert isinstance(normalized, FakePipeline)
    assert calls == ["builder"]


def test_normalize_initial_assumption_preserves_sequence_order():
    calls = []
    builders = (
        FakePipelineBuilder("first", calls),
        FakePipelineBuilder("second", calls),
    )

    normalized = normalize_initial_assumption(builders)

    assert [type(item) for item in normalized] == [FakePipeline, FakePipeline]
    assert calls == ["first", "second"]


def test_resolve_initial_assumption_plan_prefers_automl_config():
    plan = resolve_initial_assumption_plan(
        automl_config={"initial_assumption": "automl"},
        industrial_config={"initial_assumption": "industrial"},
    )

    assert plan.source is InitialAssumptionSource.AUTOML_CONFIG
    assert plan.assumption == "automl"


def test_resolve_initial_assumption_plan_uses_industrial_config_as_fallback():
    plan = resolve_initial_assumption_plan(
        automl_config={"initial_assumption": None},
        industrial_config={"initial_assumption": "industrial"},
    )

    assert plan.source is InitialAssumptionSource.INDUSTRIAL_CONFIG
    assert plan.assumption == "industrial"


def test_normalize_initial_assumption_from_configs_combines_resolution_and_building():
    calls = []
    normalized = normalize_initial_assumption_from_configs(
        automl_config={"initial_assumption": None},
        industrial_config={"initial_assumption": FakePipelineBuilder("industrial", calls)},
    )

    assert isinstance(normalized, FakePipeline)
    assert calls == ["industrial"]
