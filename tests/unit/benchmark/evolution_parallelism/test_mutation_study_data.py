import numpy as np

from benchmark.industrial.experiments.mutation_diversity.data import load_training_data
from benchmark.industrial.experiments.mutation_diversity.study import load_protocol


def test_classification_uses_only_official_train_and_disjoint_fixed_control(monkeypatch):
    import sktime.datasets as datasets
    loader = datasets.load_acsf1
    calls = []

    def guarded(*args, **kwargs):
        calls.append(kwargs['split'])
        assert kwargs['split'] == 'train'
        return loader(*args, **kwargs)

    monkeypatch.setattr(datasets, 'load_acsf1', guarded)
    data = load_training_data('classification', load_protocol())
    search, control = set(data.source['search_indices']), set(data.source['outer_indices'])
    assert calls == ['train'] and not search & control
    for fold in data.source['folds']:
        left, right = set(fold['train_indices']), set(fold['validation_indices'])
        assert not left & right and (left | right) == search
        assert not (left | right) & control
    assert data.source['official_test_used'] is False


def test_forecast_holdout_is_inside_train_and_official_test_values_cannot_change_it(monkeypatch):
    import sktime.datasets as datasets
    loader = datasets.load_airline
    first = load_training_data('ts_forecasting', load_protocol())

    def altered():
        values = loader().copy()
        values.iloc[-12:] = 1e9
        return values

    monkeypatch.setattr(datasets, 'load_airline', altered)
    second = load_training_data('ts_forecasting', load_protocol())
    assert first.source == second.source
    official_train = loader().to_numpy(dtype=float)[:-12]
    np.testing.assert_array_equal(first.refit.features.numpy().reshape(-1), official_train[:-12])
    np.testing.assert_array_equal(first.test.target.numpy().reshape(-1), official_train[-12:])
    for (left, right), (left2, right2) in zip(first.folds, second.folds):
        np.testing.assert_array_equal(left.features, left2.features)
        np.testing.assert_array_equal(right.target, right2.target)
