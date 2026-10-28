"""Final held-out evaluation, separate from candidate selection on TRAIN."""
from dataclasses import dataclass, replace
from time import perf_counter

from .clock import search_complete
from .data import load_data


@dataclass(frozen=True)
class FinalEvaluation:
    refit_seconds: float | None = None
    predict_seconds: float | None = None
    test_loss: float | None = None


def evaluate_selected(config, evolved, winner, source, root):
    # Check before requesting the held-out container, including for pilots.
    if config.pilot or not search_complete(evolved, config.generations):
        return FinalEvaluation()
    from .search import loss_value, to_numpy, write_json
    import pandas as pd

    data = load_data(config.task, config.seed, include_test=True)
    if data.source['train_sha256'] != source['train_sha256']:
        raise ValueError('TRAIN changed before the final evaluation')
    write_json(root / 'test_source.json', data.source)
    try:
        start = perf_counter()
        winner.fit(data.refit)
        refit_seconds = perf_counter() - start
        start = perf_counter()
        predicted = winner.predict(replace(data.test, target=None, state='predict'),
                                   output_mode='labels' if config.task == 'classification' else 'default')
        predict_seconds = perf_counter() - start
        test_loss = loss_value(config.task, data.test.target, predicted.predict)
        pd.DataFrame({'target': to_numpy(data.test.target).reshape(-1),
                      'prediction': to_numpy(predicted.predict).reshape(-1)}).to_csv(root / 'predictions.csv', index=False)
        return FinalEvaluation(refit_seconds, predict_seconds, test_loss)
    finally:
        winner.unfit()
