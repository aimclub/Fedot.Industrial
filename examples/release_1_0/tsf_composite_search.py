"""Run a small Industrial evolutionary search over atomic forecasting templates on Airline TRAIN."""

import argparse
from pathlib import Path

from benchmark.industrial.experiments.evolution_parallelism.config import SearchConfig
from benchmark.industrial.experiments.evolution_parallelism.search import run_search
from benchmark.industrial.experiments.mutation_diversity.study import MutationStudy
from fedot_ind.core.models.composite_templates import ARINO, CWT, HFCM


class CompositeSearch(MutationStudy):
    keep_history = True

    def load_data(self):
        import numpy as np
        from fedot import create_data
        from fedot.core.data.split.temporal import temporal_tensor_holdout
        from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
        from sktime.datasets import load_airline

        from benchmark.industrial.experiments.evolution_parallelism.data import SearchData, array_digest

        train = load_airline().iloc[:-12].to_numpy(dtype=np.float32)
        horizon = 4
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(horizon))

        def split(end):
            data = create_data(train[:end], target=train[:end], task=task, ts_forecast_horizon=horizon,
                               without_target=True, use_cache=False)
            return temporal_tensor_holdout(data)

        first_end, second_end, last_end = 108, 120, len(train)
        refit, evaluate = split(last_end)
        return SearchData(
            (split(first_end), split(second_end)), refit, evaluate,
            {"dataset": "Airline", "source": "sktime TRAIN, final 12 observations excluded",
             "train_sha256": array_digest(train), "horizon": horizon, "test_used_for_search": False,
             "inner_validation_end": [first_end, second_end], "outer_train_end": last_end},
        )

    def search_settings(self, settings):
        operations = [HFCM.operation, ARINO.operation, CWT.operation, "lagged_ridge_forecaster"]
        return {
            **settings, "population_size": 4, "candidate_seconds": 45, "max_depth": 2,
            "tasks": {self.task: {"operations": operations, "initial_chains": [[op] for op in operations]}},
            "operation_parameters": {CWT.operation: {"cwt.window": 12, "encoders.width": 4, "training.epochs": 1}},
        }

    def implementation_paths(self):
        from fedot_ind.core.models import composite_templates
        from fedot_ind.core.models.ts_forecasting import composite
        from industrial_forecasting import dwt, wavelet_hfcm
        from industrial_forecasting.neural import wavelet_attention

        return [*super().implementation_paths(), Path(__file__),
                *(Path(module.__file__) for module in (composite_templates, composite, dwt,
                                                       wavelet_hfcm, wavelet_attention))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("sequential", "dask"), default="sequential")
    args = parser.parse_args()
    task = "ts_forecasting"
    config = SearchConfig(task, args.backend, generations=2, workers=2, budget_seconds=180,
                          seed=42, lifecycle="search", pilot=True)
    run_search(config, args.output, study=CompositeSearch("structural", task))


if __name__ == "__main__":
    main()
