"""Typed configuration boundary for Industrial evolutionary optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


EVOLUTION_CONFIG_SCHEMA = "industrial_evolution_config@1"


class MutationAgentType(str, Enum):
    """Adaptive mutation agents supported by the Industrial optimiser."""

    RANDOM = "random"
    BANDIT = "bandit"
    CONTEXTUAL_BANDIT = "contextual_bandit"
    NEURAL_BANDIT = "neural_bandit"


class MutationStrategy(str, Enum):
    """Named probability profiles from the Industrial mutation catalogue."""

    PARAMS = "params_mutation_strategy"
    GROWTH = "growth_mutation_strategy"
    REGULARIZATION = "regularization_mutation_strategy"
    INITIAL_POPULATION_DIVERSITY = "initial_population_diversity_strategy"
    UNIQUE_POPULATION = "unique_population_strategy"


class EvolutionConfigViolationCode(str, Enum):
    """Stable categories emitted by the configuration boundary."""

    INVALID_PAYLOAD = "invalid_payload"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    INVALID_SECTION = "invalid_section"
    UNKNOWN_FIELD = "unknown_field"
    CONFLICTING_FIELDS = "conflicting_fields"
    INVALID_VALUE = "invalid_value"
    INVALID_TYPE = "invalid_type"


@dataclass(frozen=True)
class EvolutionConfigViolation:
    """One deterministic configuration violation."""

    code: EvolutionConfigViolationCode
    path: str
    message: str
    value: object | None = None

    def to_record(self) -> dict[str, object]:
        record = {
            "code": self.code.value,
            "path": self.path,
            "message": self.message,
            "value": self.value,
        }
        return {key: value for key, value in record.items() if value is not None}


class EvolutionConfigError(ValueError):
    """Raised once with every independent configuration violation."""

    def __init__(self, violations: tuple[EvolutionConfigViolation, ...]):
        if not violations:
            raise ValueError("EvolutionConfigError requires at least one violation")
        self.violations = violations
        details = "; ".join(
            f"{violation.path}: {violation.message}" for violation in violations
        )
        super().__init__(f"Invalid evolution configuration: {details}")


@dataclass(frozen=True)
class ResourceBudget:
    """Bounded work used while constructing and extending populations."""

    min_population_size: int = 10
    mutation_attempts_per_candidate: int = 50
    population_extension_attempts: int = 100

    def __post_init__(self) -> None:
        for field_name, value in self.to_record().items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def to_record(self) -> dict[str, int]:
        return {
            "min_population_size": self.min_population_size,
            "mutation_attempts_per_candidate": self.mutation_attempts_per_candidate,
            "population_extension_attempts": self.population_extension_attempts,
        }


@dataclass(frozen=True)
class ExecutionPolicy:
    """Execution decisions that are independent from mutation selection."""

    initial_graphs_prevalidated: bool = False
    retry_initial_population_without_timer: bool = True
    diagnostics_output_dir: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.initial_graphs_prevalidated, bool):
            raise ValueError("initial_graphs_prevalidated must be a boolean")
        if not isinstance(self.retry_initial_population_without_timer, bool):
            raise ValueError("retry_initial_population_without_timer must be a boolean")
        if self.diagnostics_output_dir is not None:
            if not isinstance(self.diagnostics_output_dir, str):
                raise ValueError("diagnostics_output_dir must be a string or null")
            if not self.diagnostics_output_dir.strip():
                raise ValueError("diagnostics_output_dir must not be empty")

    def to_record(self) -> dict[str, object]:
        return {
            "initial_graphs_prevalidated": self.initial_graphs_prevalidated,
            "retry_initial_population_without_timer": self.retry_initial_population_without_timer,
            "diagnostics_output_dir": self.diagnostics_output_dir,
        }


@dataclass(frozen=True)
class EvolutionConfig:
    """Canonical internal configuration for one evolutionary optimiser."""

    mutation_agent: MutationAgentType = MutationAgentType.RANDOM
    mutation_strategy: MutationStrategy = MutationStrategy.PARAMS
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.mutation_agent, MutationAgentType):
            raise ValueError("mutation_agent must be a MutationAgentType")
        if not isinstance(self.mutation_strategy, MutationStrategy):
            raise ValueError("mutation_strategy must be a MutationStrategy")
        if not isinstance(self.resource_budget, ResourceBudget):
            raise ValueError("resource_budget must be a ResourceBudget")
        if not isinstance(self.execution_policy, ExecutionPolicy):
            raise ValueError("execution_policy must be an ExecutionPolicy")

    def to_record(self) -> dict[str, object]:
        """Return the canonical JSON-compatible transport shape."""
        return {
            "schema_version": EVOLUTION_CONFIG_SCHEMA,
            "mutation": {
                "agent": self.mutation_agent.value,
                "strategy": self.mutation_strategy.value,
            },
            "resource_budget": self.resource_budget.to_record(),
            "execution_policy": self.execution_policy.to_record(),
        }

    def with_initial_graphs_prevalidated(self, value: bool) -> "EvolutionConfig":
        return replace(
            self,
            execution_policy=replace(
                self.execution_policy,
                initial_graphs_prevalidated=value,
            ),
        )


_ROOT_FIELDS = frozenset({
    "schema_version",
    "mutation",
    "resource_budget",
    "execution_policy",
    "mutation_agent",
    "mutation_strategy",
    "min_population_size",
    "mutation_attempts_per_candidate",
    "population_extension_attempts",
    "initial_graphs_prevalidated",
    "retry_initial_population_without_timer",
    "diagnostics_output_dir",
})
_MUTATION_FIELDS = frozenset({"agent", "strategy"})
_RESOURCE_FIELDS = frozenset(ResourceBudget().to_record())
_POLICY_FIELDS = frozenset(ExecutionPolicy().to_record())


def _mapping_section(
        payload: Mapping[str, Any],
        name: str,
        violations: list[EvolutionConfigViolation],
) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if isinstance(value, Mapping):
        return value
    violations.append(EvolutionConfigViolation(
        code=EvolutionConfigViolationCode.INVALID_SECTION,
        path=name,
        message="must be an object",
        value=value,
    ))
    return {}


def _collect_unknown_fields(
        payload: Mapping[str, Any],
        allowed: frozenset[str],
        path: str,
        violations: list[EvolutionConfigViolation],
) -> None:
    for name in sorted(set(payload) - allowed):
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.UNKNOWN_FIELD,
            path=f"{path}.{name}" if path else name,
            message="is not supported",
            value=payload[name],
        ))


def _enum_value(
        enum_type,
        value: object,
        path: str,
        violations: list[EvolutionConfigViolation],
):
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        supported = ", ".join(item.value for item in enum_type)
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.INVALID_VALUE,
            path=path,
            message=f"must be one of: {supported}",
            value=value,
        ))
        return next(iter(enum_type))


def _positive_integer(
        value: object,
        path: str,
        violations: list[EvolutionConfigViolation],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.INVALID_TYPE,
            path=path,
            message="must be a positive integer",
            value=value,
        ))
        return 1
    return value


def _boolean(
        value: object,
        path: str,
        violations: list[EvolutionConfigViolation],
) -> bool:
    if not isinstance(value, bool):
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.INVALID_TYPE,
            path=path,
            message="must be a boolean",
            value=value,
        ))
        return False
    return value


def _optional_path(
        value: object,
        path: str,
        violations: list[EvolutionConfigViolation],
) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)) and str(value).strip():
        return str(value)
    violations.append(EvolutionConfigViolation(
        code=EvolutionConfigViolationCode.INVALID_TYPE,
        path=path,
        message="must be a non-empty path string or null",
        value=value,
    ))
    return None


def _section_or_legacy_value(
        section: Mapping[str, Any],
        section_key: str,
        payload: Mapping[str, Any],
        legacy_key: str,
        default: object,
        path: str,
        violations: list[EvolutionConfigViolation],
) -> object:
    has_section_value = section_key in section
    has_legacy_value = legacy_key in payload
    if has_section_value and has_legacy_value and section[section_key] != payload[legacy_key]:
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.CONFLICTING_FIELDS,
            path=path,
            message=f"conflicts with legacy field {legacy_key}",
            value=section[section_key],
        ))
    if has_section_value:
        return section[section_key]
    return payload.get(legacy_key, default)


def normalize_evolution_config(
        payload: EvolutionConfig | Mapping[str, Any] | None,
) -> EvolutionConfig:
    """Normalize legacy or canonical input exactly once at the boundary."""
    if payload is None:
        return EvolutionConfig()
    if isinstance(payload, EvolutionConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise EvolutionConfigError((EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.INVALID_PAYLOAD,
            path="$",
            message="must be an object",
            value=payload,
        ),))

    violations: list[EvolutionConfigViolation] = []
    _collect_unknown_fields(payload, _ROOT_FIELDS, "", violations)
    schema_version = payload.get("schema_version")
    if schema_version not in (None, EVOLUTION_CONFIG_SCHEMA):
        violations.append(EvolutionConfigViolation(
            code=EvolutionConfigViolationCode.UNSUPPORTED_SCHEMA,
            path="schema_version",
            message=f"must equal {EVOLUTION_CONFIG_SCHEMA}",
            value=schema_version,
        ))

    mutation = _mapping_section(payload, "mutation", violations)
    resources = _mapping_section(payload, "resource_budget", violations)
    policy = _mapping_section(payload, "execution_policy", violations)
    _collect_unknown_fields(mutation, _MUTATION_FIELDS, "mutation", violations)
    _collect_unknown_fields(resources, _RESOURCE_FIELDS, "resource_budget", violations)
    _collect_unknown_fields(policy, _POLICY_FIELDS, "execution_policy", violations)

    defaults = EvolutionConfig()
    agent = _enum_value(
        MutationAgentType,
        _section_or_legacy_value(
            mutation,
            "agent",
            payload,
            "mutation_agent",
            defaults.mutation_agent.value,
            "mutation.agent",
            violations,
        ),
        "mutation.agent",
        violations,
    )
    strategy = _enum_value(
        MutationStrategy,
        _section_or_legacy_value(
            mutation,
            "strategy",
            payload,
            "mutation_strategy",
            defaults.mutation_strategy.value,
            "mutation.strategy",
            violations,
        ),
        "mutation.strategy",
        violations,
    )
    resource_budget = ResourceBudget(
        min_population_size=_positive_integer(
            _section_or_legacy_value(
                resources,
                "min_population_size",
                payload,
                "min_population_size",
                defaults.resource_budget.min_population_size,
                "resource_budget.min_population_size",
                violations,
            ),
            "resource_budget.min_population_size",
            violations,
        ),
        mutation_attempts_per_candidate=_positive_integer(
            _section_or_legacy_value(
                resources,
                "mutation_attempts_per_candidate",
                payload,
                "mutation_attempts_per_candidate",
                defaults.resource_budget.mutation_attempts_per_candidate,
                "resource_budget.mutation_attempts_per_candidate",
                violations,
            ),
            "resource_budget.mutation_attempts_per_candidate",
            violations,
        ),
        population_extension_attempts=_positive_integer(
            _section_or_legacy_value(
                resources,
                "population_extension_attempts",
                payload,
                "population_extension_attempts",
                defaults.resource_budget.population_extension_attempts,
                "resource_budget.population_extension_attempts",
                violations,
            ),
            "resource_budget.population_extension_attempts",
            violations,
        ),
    )
    execution_policy = ExecutionPolicy(
        initial_graphs_prevalidated=_boolean(
            _section_or_legacy_value(
                policy,
                "initial_graphs_prevalidated",
                payload,
                "initial_graphs_prevalidated",
                defaults.execution_policy.initial_graphs_prevalidated,
                "execution_policy.initial_graphs_prevalidated",
                violations,
            ),
            "execution_policy.initial_graphs_prevalidated",
            violations,
        ),
        retry_initial_population_without_timer=_boolean(
            _section_or_legacy_value(
                policy,
                "retry_initial_population_without_timer",
                payload,
                "retry_initial_population_without_timer",
                defaults.execution_policy.retry_initial_population_without_timer,
                "execution_policy.retry_initial_population_without_timer",
                violations,
            ),
            "execution_policy.retry_initial_population_without_timer",
            violations,
        ),
        diagnostics_output_dir=_optional_path(
            _section_or_legacy_value(
                policy,
                "diagnostics_output_dir",
                payload,
                "diagnostics_output_dir",
                defaults.execution_policy.diagnostics_output_dir,
                "execution_policy.diagnostics_output_dir",
                violations,
            ),
            "execution_policy.diagnostics_output_dir",
            violations,
        ),
    )
    if violations:
        raise EvolutionConfigError(tuple(violations))
    return EvolutionConfig(
        mutation_agent=agent,
        mutation_strategy=strategy,
        resource_budget=resource_budget,
        execution_policy=execution_policy,
    )
