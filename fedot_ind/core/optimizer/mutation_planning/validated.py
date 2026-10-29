"""Ephemeral validated outcomes, owned by one immutable mutation enumeration."""

from dataclasses import dataclass

from .contracts import AppliedMutation, MutationCandidates, MutationLimits, MutationPlan, RawInputSpec
from .planning import _enumerate_mutation_plans
from fedot_ind.core.optimizer.graph_validation.contracts import GraphSpec


@dataclass(frozen=True)
class ValidatedEnumeration:
    candidates: MutationCandidates
    source: GraphSpec
    limits: MutationLimits
    raw_input: RawInputSpec | None
    outcomes: tuple[tuple[MutationPlan, AppliedMutation], ...]

    def outcome_for(self, plan, spec, limits, raw_input):
        # Equal-looking plans from a different enumeration have no local proof.
        if spec is not self.source or limits != self.limits or raw_input != self.raw_input:
            return None
        return next((outcome for owner, outcome in self.outcomes if owner is plan), None)


def enumerate_validated_mutation_plans(spec, operations, *, limits, kinds, raw_input=None, catalog_offset=0):
    outcomes = []
    candidates = _enumerate_mutation_plans(spec, operations, limits=limits, kinds=kinds,
                                           raw_input=raw_input, catalog_offset=catalog_offset,
                                           _validated_outcomes=outcomes)
    return ValidatedEnumeration(candidates, spec, limits, raw_input, tuple(outcomes))
