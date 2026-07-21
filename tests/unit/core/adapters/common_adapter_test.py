import pytest
import pandas as pd

from fedot_ind.core.adapters.common_adapter.contracts import AutoMLBudget
from fedot_ind.core.adapters.common_adapter.fake_automl import FakeAutoMLBackend


def test_fake_backend_univariate():
    adapter = FakeAutoMLBackend(supports_quantiles=True)

    assert adapter.result is None

    data = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    metadata = {'mode': 'univariate'}
    budget = AutoMLBudget(
        time_limit=10.0,
        trial_limit=5,
        memory_hint=1024,
        random_seed=42
    )

    adapter.fit(data, metadata, budget)

    assert adapter.result is not None
    assert adapter.result.wall_clock > 0
    assert adapter.result.fitted_model_count == 1
    assert adapter.result.failure_count == 0
    assert adapter.result.quantile_support

    predictions = adapter.predict(data, horizon=3)
    assert len(predictions) == 3
    assert (predictions['prediction'] == 5).all()

    quantiles = adapter.predict_quantiles(
        data, horizon=2, quantiles=[0.1, 0.5, 0.9])
    assert len(quantiles) == 6

    avail = adapter.availability()
    assert avail['univariate']
    assert avail['quantiles']

    resources = adapter.resource_report()
    assert 'memory_mb' in resources
    assert 'cpu_cores' in resources


def test_fake_backend_panel_mode():
    adapter = FakeAutoMLBackend()

    data = pd.DataFrame({
        'series_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    metadata = {'mode': 'panel'}
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)

    adapter.fit(data, metadata, budget)

    assert adapter.result.fitted_model_count == 1
    assert adapter.result.failure_count == 0
    assert adapter.result.quantile_support

    predictions = adapter.predict(data, horizon=2)
    assert len(predictions) == 4

    quantiles = adapter.predict_quantiles(
        data, horizon=2, quantiles=[0.1, 0.9])
    assert len(quantiles) == 8
    assert set(quantiles['series_id']) == {'A', 'B'}


def test_fake_backend_multivariate_mode():
    adapter = FakeAutoMLBackend()

    data = pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'feature1': [2, 3, 4, 5, 6],
        'feature2': [3, 4, 5, 6, 7]
    })
    metadata = {'mode': 'multivariate'}
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)

    adapter.fit(data, metadata, budget, target_column='target')

    assert adapter.result.fitted_model_count == 1

    predictions = adapter.predict(data, horizon=3)
    assert len(predictions) == 3


def test_quantile_support_flag():
    adapter = FakeAutoMLBackend(supports_quantiles=False)

    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    metadata = {'mode': 'univariate'}
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)
    adapter.fit(data, metadata, budget)
    assert adapter.result.quantile_support is False
    with pytest.raises(NotImplementedError):
        adapter.predict_quantiles(data, horizon=2, quantiles=[0.1, 0.5, 0.9])

    adapter2 = FakeAutoMLBackend(supports_quantiles=True)
    adapter2.fit(data, metadata, budget)
    assert adapter2.result.quantile_support
    result = adapter2.predict_quantiles(
        data, horizon=2, quantiles=[0.1, 0.5, 0.9])
    assert len(result) == 6


def test_fake_backend_failure_count():
    adapter = FakeAutoMLBackend(fail_on_fit=True)

    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    metadata = {'mode': 'univariate'}
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)

    with pytest.raises(RuntimeError):
        adapter.fit(data, metadata, budget)

    assert adapter.failure_count == 1


def test_fake_backend_panel_failure_count_is_not_duplicated():
    adapter = FakeAutoMLBackend(fail_on_fit=True)

    data = pd.DataFrame({
        'series_id': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)

    with pytest.raises(RuntimeError):
        adapter.fit(data, {'mode': 'panel'}, budget)

    assert adapter.failure_count == 1


def test_fake_backend_rejects_unknown_mode():
    adapter = FakeAutoMLBackend()
    data = pd.DataFrame({'value': [1, 2, 3]})
    budget = AutoMLBudget(time_limit=10.0, trial_limit=5,
                          memory_hint=1024, random_seed=42)

    with pytest.raises(ValueError, match='unsupported is not supported'):
        adapter.fit(data, {'mode': 'unsupported'}, budget)

    assert adapter.failure_count == 1


@pytest.mark.parametrize(
    'budget_kwargs',
    [
        {'time_limit': 0.0, 'trial_limit': 1, 'memory_hint': 128, 'random_seed': 1},
        {'time_limit': 1.0, 'trial_limit': -1,
            'memory_hint': 128, 'random_seed': 1},
        {'time_limit': 1.0, 'trial_limit': 1, 'memory_hint': 0, 'random_seed': 1},
    ]
)
def test_budget_validates_resource_bounds(budget_kwargs):
    with pytest.raises(ValueError):
        AutoMLBudget(**budget_kwargs)
