import json
from dataclasses import replace

from hypothesis import given, settings, strategies as st
import pytest

from fedot_ind.core.optimizer.mutation_planning import (
    AppliedMutation, CandidateStatus, MutationKind, MutationRejectionCode,
    ReplaceOperation, enumerate_mutation_plans, apply_mutation_plan,
)
from .test_planning import CATALOG, LIMITS, capability, chain


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 40), st.integers(0, 200), st.sampled_from(tuple(MutationKind)))
def test_budget_and_diagnostic_conservation(budget, offset, kind):
    source = chain()
    result = enumerate_mutation_plans(source, CATALOG, limits=replace(LIMITS, max_candidates=budget),
                                      kinds=(kind,), catalog_offset=offset)
    assert result.examined == len(result.diagnostics) <= budget
    assert sum(item.status is CandidateStatus.ADMISSIBLE for item in result.diagnostics) == len(result.plans)
    assert all(item.reason_codes for item in result.diagnostics if item.status is not CandidateStatus.ADMISSIBLE)
    assert source == chain()
    assert result == enumerate_mutation_plans(source, CATALOG, limits=replace(LIMITS, max_candidates=budget),
                                              kinds=(kind,), catalog_offset=offset)
    record = result.diagnostics_record()
    assert json.loads(json.dumps(record)) == record
    assert sum(record['status_counts'].values()) == result.examined
    for plan in result.plans:
        assert isinstance(apply_mutation_plan(source, plan, limits=LIMITS), AppliedMutation)


def test_disabled_families_do_not_consume_budget():
    result = enumerate_mutation_plans(chain(), tuple(capability(str(i)) for i in range(100)),
                                      limits=replace(LIMITS, max_candidates=1), kinds=(MutationKind.DROP,))
    assert result.examined == 1 and len(result.plans) == 1
    assert result.diagnostics_record()['family_counts'] == {'drop': 1}
    empty = enumerate_mutation_plans(chain(), CATALOG, limits=LIMITS, kinds=())
    assert empty.examined == 0 and not empty.truncated


def test_small_budget_covers_locations_instead_of_exhausting_the_first_node():
    spec = chain(4)
    result = enumerate_mutation_plans(spec, tuple(capability(str(i)) for i in range(20)),
                                      limits=replace(LIMITS, max_candidates=5), kinds=(MutationKind.REPLACE,))
    assert {item.edit.node_id for item in result.diagnostics} == set(spec.node_ids)


def test_rotation_exposes_late_catalog_entries_without_unbounded_scan():
    catalog = tuple(capability(str(i)) for i in range(100))
    first = enumerate_mutation_plans(chain(1), catalog, limits=replace(LIMITS, max_candidates=1),
                                     kinds=(MutationKind.REPLACE,), catalog_offset=0)
    late = enumerate_mutation_plans(chain(1), catalog, limits=replace(LIMITS, max_candidates=1),
                                    kinds=(MutationKind.REPLACE,), catalog_offset=99)
    assert first.diagnostics[0].edit.operation.name == '0'
    assert late.diagnostics[0].edit.operation.name == '99'
    assert late.examined == 1 and late.truncated
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            enumerate_mutation_plans(chain(), catalog, limits=LIMITS, catalog_offset=value)


def test_diagnostics_name_kind_target_type_and_unchanged_rejections():
    spec = chain(1)
    result = enumerate_mutation_plans(spec, (spec.nodes[0].capabilities, *CATALOG), limits=LIMITS)
    reasons = result.diagnostics_record()['reason_counts']
    assert reasons[MutationRejectionCode.SAME_OPERATION.value] > 0
    assert reasons[MutationRejectionCode.OPERATION_KIND.value] > 0
    assert all(isinstance(item.edit, ReplaceOperation) for item in result.diagnostics
               if item.family == 'replace')
    wrong_type = capability('image', input_data_types=('image',), output_data_types=('image',))
    rejected = enumerate_mutation_plans(spec, (wrong_type,), limits=LIMITS, kinds=(MutationKind.REPLACE,))
    assert rejected.diagnostics_record()['reason_counts'][MutationRejectionCode.DATA_TYPE.value] > 0
