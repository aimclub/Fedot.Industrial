from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
from benchmark.industrial.experiments.evolution_parallelism import final_evaluation as module


@pytest.mark.parametrize('pilot,evolved', [(True, 15), (False, 14)])
def test_ineligible_search_never_loads_test(monkeypatch, tmp_path, pilot, evolved):
    monkeypatch.setattr(module, 'load_data', lambda *a, **k: pytest.fail('TEST requested'))
    result = module.evaluate_selected(SearchConfig('regression', 'sequential', pilot=pilot),
                                      evolved, None, {}, tmp_path)
    assert result == module.FinalEvaluation()
    assert not list(tmp_path.iterdir())


@dataclass
class Data:
    target: object
    state: str = 'fit'


@pytest.mark.parametrize('failure', [None, 'fit', 'predict'])
def test_final_boundary_hides_target_and_always_unfits(monkeypatch, tmp_path, failure):
    source = {'train_sha256': 'frozen'}
    data = SimpleNamespace(source=source, refit=object(), test=Data(np.array([1., 2.])))
    monkeypatch.setattr(module, 'load_data', lambda *a, **k: data)
    calls = []

    class Winner:
        def fit(self, value):
            calls.append('fit')
            assert value is data.refit
            if failure == 'fit':
                raise RuntimeError('fit')

        def predict(self, value, **kwargs):
            calls.append('predict')
            assert value.target is None and value.state == 'predict'
            if failure == 'predict':
                raise RuntimeError('predict')
            return SimpleNamespace(predict=np.array([1., 2.]))

        def unfit(self):
            calls.append('unfit')

    def run():
        return module.evaluate_selected(SearchConfig('regression', 'sequential'), 15, Winner(), source, tmp_path)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        assert run().test_loss == 0
        assert (tmp_path / 'predictions.csv').is_file()
    assert calls[-1] == 'unfit'
    np.testing.assert_array_equal(data.test.target, [1., 2.])
