from dataclasses import replace
import random

import pytest

from fedot_ind.core.optimizer.mutation_planning import MutationKind, apply_mutation_plan, enumerate_mutation_plans
from fedot_ind.core.optimizer.mutation_planning.validated import enumerate_validated_mutation_plans
from tests.unit.core.optimizer.mutation_planning.test_planning import chain, CATALOG, LIMITS


@pytest.mark.parametrize('offset', [0, 1, 42, 123])
def test_same_order_outcomes_diagnostics_and_rng(offset):
    spec = chain()
    kinds = (MutationKind.REPLACE, MutationKind.ADD, MutationKind.DROP)
    state = random.getstate()
    expected = enumerate_mutation_plans(spec, CATALOG, limits=LIMITS, kinds=kinds, catalog_offset=offset)
    actual = enumerate_validated_mutation_plans(spec, CATALOG, limits=LIMITS, kinds=kinds, catalog_offset=offset)
    assert actual.candidates == expected
    assert actual.candidates.diagnostics_record() == expected.diagnostics_record()
    assert random.getstate() == state
    for plan in actual.candidates.plans:
        assert actual.outcome_for(plan, spec, LIMITS, None) == apply_mutation_plan(spec, plan, limits=LIMITS)
        assert actual.outcome_for(replace(plan), spec, LIMITS, None) is None
        assert actual.outcome_for(plan, replace(spec), LIMITS, None) is None
        assert actual.outcome_for(plan, spec, replace(LIMITS, max_depth=1), None) is None


def test_outcome_lookup_does_not_apply_or_validate_again(monkeypatch):
    from fedot_ind.core.optimizer.mutation_planning import planning
    spec = chain()
    enumeration = enumerate_validated_mutation_plans(spec, CATALOG, limits=LIMITS, kinds=(MutationKind.ADD,))

    def forbidden(*args, **kwargs):
        raise AssertionError('A locally proved outcome must not be recomputed')

    monkeypatch.setattr(planning, 'apply_mutation_plan', forbidden)
    assert enumeration.candidates.plans
    for plan in enumeration.candidates.plans:
        assert enumeration.outcome_for(plan, spec, LIMITS, None) is not None


@pytest.mark.parametrize('seed', [42, 123, 2026])
def test_runtime_reuse_preserves_graph_decision_and_rng_against_revalidation(monkeypatch, seed):
    from fedot_ind.core.optimizer import mutation_runtime
    from fedot_ind.core.optimizer.randomness import operator_random_scope
    from tests.unit.core.optimizer.test_mutation_runtime import _planned
    from fedot_ind.core.optimizer.mutation_planning.validated import ValidatedEnumeration

    operator, parent = _planned()
    with operator_random_scope(seed):
        candidate = operator(parent)
        decision = operator.last_execution.to_record()
        state = random.getstate()

    def fresh(self, plan, spec, limits, raw_input):
        return apply_mutation_plan(spec, plan, limits=limits, raw_input=raw_input)

    monkeypatch.setattr(ValidatedEnumeration, 'outcome_for', fresh)
    reference_operator, reference_parent = _planned()
    with operator_random_scope(seed):
        reference = reference_operator(reference_parent)
        assert random.getstate() == state
        assert reference_operator.last_execution.to_record() == decision
    assert mutation_runtime.project_for_mutation(candidate.graph, operator.graph_generation_params) == \
        mutation_runtime.project_for_mutation(reference.graph, reference_operator.graph_generation_params)
