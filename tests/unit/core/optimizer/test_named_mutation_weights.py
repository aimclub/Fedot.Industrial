from types import SimpleNamespace
import pytest

from fedot_ind.core.optimizer.mutation import mutation_weights


def actions(*names):
    return [SimpleNamespace(name=name) for name in names]


def test_missing_parameter_action_explains_the_current_degenerate_profile():
    assert mutation_weights(actions('single_change', 'single_add', 'single_drop'), [.7, .3, 0., 0., 0.]) == [1., 0., 0.]


def test_named_structural_weights_follow_action_names_not_positions():
    profile = dict(parameter_change=0., replace=.4, drop=.2, add=.4, edge=0.)
    assert mutation_weights(actions('single_drop', 'single_change', 'single_add'), profile) == [.2, .4, .4]
    assert mutation_weights(actions('single_change', 'single_drop'), profile) == pytest.approx([2 / 3, 1 / 3])


def test_legacy_drop_add_edge_contract_is_explicit():
    assert mutation_weights(actions('single_edge', 'single_add', 'single_drop'), [0, 0, 1, 2, 3]) == [.5, 1 / 3, 1 / 6]


@pytest.mark.parametrize('bad', [-1, True, float('nan'), float('inf')])
def test_invalid_probabilities_are_not_silently_accepted(bad):
    with pytest.raises(ValueError):
        mutation_weights(actions('single_add'), [0, 0, 0, bad, 0])
