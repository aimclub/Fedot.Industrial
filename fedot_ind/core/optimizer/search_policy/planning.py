"""Pure decisions: elapsed-budget stage and quality-preserving family slots."""
from dataclasses import dataclass
from math import ceil, isfinite

from .contracts import number


def stage_for_budget(policy, elapsed_seconds, budget_seconds):
    number(elapsed_seconds, 'elapsed_seconds')
    number(budget_seconds, 'budget_seconds')
    if not budget_seconds:
        raise ValueError('A staged search requires a positive time budget')
    fraction = min(1., elapsed_seconds / budget_seconds)
    return next((stage for stage in policy.stages if fraction < stage.until), policy.stages[-1])


def action_weights(actions, stage):
    from fedot_ind.core.optimizer.mutation import mutation_name
    kinds = {'parameter_change_mutation': 'parameter_change', 'rewrite_subgraph_mutation': 'rewrite',
             'single_change': 'replace', 'simple': 'replace', 'single_add': 'add', 'growth': 'add',
             'local_growth': 'add', 'tree_growth': 'add', 'single_drop': 'drop', 'reduce': 'drop'}
    names = [mutation_name(action) for action in actions]
    if any(name not in kinds for name in names):
        raise ValueError('Staged search does not replace arbitrary custom mutation policies')
    weights = stage.weights.to_record()
    result = [weights[kinds[name]] for name in names]
    total = sum(result)
    if not total:
        raise ValueError('A stage has no positive configured mutation action')
    return [value / total for value in result]


@dataclass(frozen=True)
class FamilyCandidate:
    key: str
    family: str
    loss: float
    nodes: int


def select_survivors(candidates, capacity, family_fraction):
    """Keep the global minimum, reserve distinct families, then fill by quality."""
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError('Survivor capacity must be positive')
    number(family_fraction, 'family_fraction', maximum=1.)
    ordered = sorted((c for c in candidates if isfinite(c.loss)), key=lambda c: (c.loss, c.nodes, c.key))
    unique = list({c.key: c for c in reversed(ordered)}.values())
    unique.sort(key=lambda c: (c.loss, c.nodes, c.key))
    if not unique:
        return ()
    selected, families = [unique[0]], {unique[0].family}
    reserved = max(1, ceil(capacity * family_fraction))
    for candidate in unique[1:]:
        if len(selected) >= min(reserved, capacity):
            break
        if candidate.family not in families:
            selected.append(candidate)
            families.add(candidate.family)
    keys = {candidate.key for candidate in selected}
    selected.extend(candidate for candidate in unique if candidate.key not in keys)
    return tuple(candidate.key for candidate in selected[:capacity])
