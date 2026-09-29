from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fedot_ind.core.optimizer.configuration import (
    EVOLUTION_CONFIG_SCHEMA,
    EvolutionConfig,
    EvolutionConfigError,
    EvolutionConfigViolation,
    EvolutionConfigViolationCode,
    ExecutionPolicy,
    MutationAgentType,
    MutationStrategy,
    ResourceBudget,
    normalize_evolution_config,
)


@st.composite
def evolution_configs(draw):
    diagnostics_dir = draw(st.one_of(
        st.none(),
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-",
            min_size=1,
            max_size=40,
        ),
    ))
    return EvolutionConfig(
        mutation_agent=draw(st.sampled_from(tuple(MutationAgentType))),
        mutation_strategy=draw(st.sampled_from(tuple(MutationStrategy))),
        resource_budget=ResourceBudget(
            min_population_size=draw(st.integers(min_value=1, max_value=100)),
            mutation_attempts_per_candidate=draw(st.integers(min_value=1, max_value=100)),
            population_extension_attempts=draw(st.integers(min_value=1, max_value=500)),
        ),
        execution_policy=ExecutionPolicy(
            initial_graphs_prevalidated=draw(st.booleans()),
            retry_initial_population_without_timer=draw(st.booleans()),
            diagnostics_output_dir=diagnostics_dir,
        ),
    )


def test_legacy_flat_configuration_normalizes_to_canonical_contract():
    config = normalize_evolution_config({
        "mutation_agent": "bandit",
        "mutation_strategy": "growth_mutation_strategy",
        "min_population_size": 7,
        "mutation_attempts_per_candidate": 11,
        "population_extension_attempts": 13,
        "initial_graphs_prevalidated": True,
        "retry_initial_population_without_timer": False,
        "diagnostics_output_dir": "results/diagnostics",
    })

    assert config.mutation_agent is MutationAgentType.BANDIT
    assert config.mutation_strategy is MutationStrategy.GROWTH
    assert config.resource_budget == ResourceBudget(7, 11, 13)
    assert config.execution_policy == ExecutionPolicy(
        initial_graphs_prevalidated=True,
        retry_initial_population_without_timer=False,
        diagnostics_output_dir="results/diagnostics",
    )
    assert config.to_record()["schema_version"] == EVOLUTION_CONFIG_SCHEMA


def test_configuration_boundary_accumulates_independent_violations():
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({
            "schema_version": "unsupported",
            "mutation": {"agent": "missing", "unknown": 1},
            "resource_budget": {"min_population_size": 0},
            "execution_policy": {"initial_graphs_prevalidated": "false"},
            "unexpected": True,
        })

    paths = [violation.path for violation in error.value.violations]
    assert paths == [
        "unexpected",
        "schema_version",
        "mutation.unknown",
        "mutation.agent",
        "resource_budget.min_population_size",
        "execution_policy.initial_graphs_prevalidated",
    ]


def test_configuration_rejects_non_mapping_payload():
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config("random")

    assert error.value.violations[0].path == "$"


def test_configuration_defaults_are_explicit_and_serializable():
    config = normalize_evolution_config(None)
    violation = EvolutionConfigViolation(
        code=EvolutionConfigViolationCode.INVALID_VALUE,
        path="mutation.agent",
        message="invalid",
    )

    assert config == EvolutionConfig()
    assert violation.to_record() == {
        "code": "invalid_value",
        "path": "mutation.agent",
        "message": "invalid",
    }
    with pytest.raises(ValueError, match="at least one violation"):
        EvolutionConfigError(())


def test_configuration_rejects_conflicting_legacy_and_canonical_fields():
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({
            "mutation_agent": "random",
            "mutation": {"agent": "bandit"},
        })

    violation = error.value.violations[0]
    assert violation.code.value == "conflicting_fields"
    assert violation.path == "mutation.agent"


def test_configuration_rejects_non_mapping_sections_and_empty_paths():
    with pytest.raises(EvolutionConfigError) as error:
        normalize_evolution_config({
            "mutation": "random",
            "resource_budget": [],
            "execution_policy": {"diagnostics_output_dir": ""},
        })

    assert [violation.path for violation in error.value.violations] == [
        "mutation",
        "resource_budget",
        "execution_policy.diagnostics_output_dir",
    ]


def test_normalizing_typed_configuration_is_idempotent():
    config = EvolutionConfig()

    assert normalize_evolution_config(config) is config


@given(evolution_configs())
def test_configuration_canonical_round_trip(config):
    payload = json.loads(json.dumps(config.to_record()))

    assert normalize_evolution_config(payload) == config


@given(evolution_configs())
def test_configuration_normalization_is_deterministic(config):
    payload = config.to_record()

    assert normalize_evolution_config(payload) == normalize_evolution_config(payload)


@given(evolution_configs())
def test_legacy_and_canonical_shapes_are_semantically_equivalent(config):
    legacy = {
        "mutation_agent": config.mutation_agent.value,
        "mutation_strategy": config.mutation_strategy.value,
        **config.resource_budget.to_record(),
        **config.execution_policy.to_record(),
    }

    assert normalize_evolution_config(legacy) == normalize_evolution_config(config.to_record())


@pytest.mark.parametrize(
    "resource_budget",
    [
        {"min_population_size": 0},
        {"mutation_attempts_per_candidate": True},
        {"population_extension_attempts": 1.5},
    ],
)
def test_resource_budget_rejects_non_positive_or_non_integer_values(resource_budget):
    payload = {"resource_budget": resource_budget}

    with pytest.raises(EvolutionConfigError):
        normalize_evolution_config(payload)


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: ResourceBudget(min_population_size=0), "positive integer"),
        (lambda: ExecutionPolicy(initial_graphs_prevalidated="false"), "must be a boolean"),
        (lambda: ExecutionPolicy(retry_initial_population_without_timer=1), "must be a boolean"),
        (lambda: ExecutionPolicy(diagnostics_output_dir=1), "must be a string"),
        (lambda: ExecutionPolicy(diagnostics_output_dir=" "), "must not be empty"),
        (lambda: EvolutionConfig(mutation_agent="random"), "MutationAgentType"),
        (lambda: EvolutionConfig(mutation_strategy="params_mutation_strategy"), "MutationStrategy"),
        (lambda: EvolutionConfig(resource_budget={}), "ResourceBudget"),
        (lambda: EvolutionConfig(execution_policy={}), "ExecutionPolicy"),
    ],
)
def test_direct_typed_construction_rejects_invalid_values(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()
