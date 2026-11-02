import json
from types import SimpleNamespace
import pytest

import numpy as np

from benchmark.industrial.experiments.mutation_diversity.study import MutationStudy, load_protocol


def test_profiles_are_opt_in_and_split_seed_is_not_the_search_seed():
    from fedot_ind.core.optimizer.configuration import EvolutionConfig, MutationStrategy
    p = load_protocol()
    assert EvolutionConfig().mutation_strategy is MutationStrategy.PARAMS
    assert MutationStudy('structural', 'classification').strategy is MutationStrategy.STRUCTURAL_BALANCED
    assert p['split_seed'] == 42 and p['seeds'] == [42, 123, 2026]
    assert p['workers'] == 4 and p['budget_seconds'] == 300


def test_mutation_observer_records_current_agent_probabilities_without_replacing_diagnostics(tmp_path):
    observed = []
    agent = SimpleNamespace(available_actions=[SimpleNamespace(name='single_change'), SimpleNamespace(name='single_add')],
                            get_action_probs=lambda value: np.array([.4, .6]))
    mutation = SimpleNamespace(observer=lambda event: observed.append(event), operator=SimpleNamespace(agent=agent))
    optimizer = SimpleNamespace(mutation=mutation)
    MutationStudy('structural', 'classification').observe_mutations(optimizer, tmp_path)
    event = SimpleNamespace(to_record=lambda: {'action': 'single_add', 'status': 'accepted'})
    mutation.observer(event)
    assert observed == [event]
    row = json.loads((tmp_path / 'mutations.jsonl').read_text('utf-8'))
    assert row['agent_probabilities'] == {'single_change': .4, 'single_add': .6}


@pytest.mark.parametrize('fail', [False, True])
def test_study_final_evaluation_hides_control_labels_and_unfits_on_error(tmp_path, fail):
    from benchmark.industrial.experiments.mutation_diversity.data import load_training_data
    data = load_training_data('ts_forecasting', load_protocol())
    calls = []

    class Winner:
        def fit(self, input_data):
            calls.append('fit')
            assert input_data is data.refit

        def predict(self, input_data, **kwargs):
            assert input_data.target is None
            if fail:
                raise RuntimeError('prediction failure')
            return SimpleNamespace(predict=data.test.target)

        def unfit(self):
            calls.append('unfit')

    if fail:
        with pytest.raises(RuntimeError, match='prediction failure'):
            MutationStudy('baseline', 'ts_forecasting').evaluate_selected(Winner(), data, tmp_path)
    else:
        result, outer = MutationStudy('baseline', 'ts_forecasting').evaluate_selected(Winner(), data, tmp_path)
        assert outer == 0 and result.test_loss is None
        assert (tmp_path / 'outer_train_predictions.csv').is_file()
    assert calls == ['fit', 'unfit']
