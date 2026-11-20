"""Small time-ordered comparison; the final Airline season is never evaluated."""

import argparse
import json

import numpy as np

from industrial_forecasting.dwt import ArinoSpec, predict_arino, predict_jin_kim
from benchmark.industrial.experiments.tsf_comparison import (
    AIRLINE_EXCLUDED_TAIL, ForecastComparisonPlan, SourceFingerprint, comparison_provenance, comparison_report,
    forecast_rows, load_airline_train,
)
from industrial_forecasting.metrics import seasonal_repeat
from industrial_forecasting.neural.page_projection import fit_page_projection
from industrial_forecasting.neural.wavelet_attention import WaveletAttentionSpec, fit_wavelet_attention
from industrial_forecasting.page_models import fit_page_mssa
from industrial_forecasting.wavelet_hfcm import fit_wavelet_hfcm
from industrial_forecasting.wire import predict_positional_wire


def _ridge(history, horizon, *, mode="none"):
    return predict_positional_wire(history, horizon, mode=mode, lags=12, frequencies=(3., 6.), widths=.75, centers=1.)


def _mssa_neural(history, *, learn_projection):
    import torch

    model = fit_page_projection(history, length=12, rank=2, baseline_lag=1,
                                learn_projection=learn_projection, epochs=8)
    with torch.inference_mode():
        return model(torch.as_tensor(history[-11:][None], dtype=torch.float32)).numpy()


def compare_on_airline_train():
    train = load_airline_train()
    multi_step = {
        "seasonal_repeat": lambda x, h: seasonal_repeat(x, h, 12),
        "ridge_lag12": _ridge,
        "wavelet_hfcm": lambda x, h: fit_wavelet_hfcm(x).predict(h),
        "arino_two_branch_control": lambda x, h: predict_arino(x, h, ArinoSpec(padding=8)),
        "jin_kim_approximation": lambda x, h: predict_jin_kim(x, h, first_end=88),
        "jin_kim_with_first_detail": lambda x, h: predict_jin_kim(x, h, first_end=88, details=(1,)),
        "fourier_position": lambda x, h: _ridge(x, h, mode="fourier"),
        "wire_position": lambda x, h: _ridge(x, h, mode="wire"),
        "cwt_attention": lambda x, h: fit_wavelet_attention(
            (x,), WaveletAttentionSpec(window=24, horizon=h, attention=True), epochs=3,
        ).predict(x),
        "cwt_uniform": lambda x, h: fit_wavelet_attention(
            (x,), WaveletAttentionSpec(window=24, horizon=h, attention=False), epochs=3,
        ).predict(x),
    }
    page = {
        "ridge_lag11": lambda x, h: predict_positional_wire(x, h, lags=11, mode="none", frequencies=(), widths=0.),
        "mssa_reference": lambda x, _: fit_page_mssa(x, length=12, rank=2).predict_block_end(x[-11:]),
        "mssa_frozen_projection": lambda x, _: _mssa_neural(x, learn_projection=False),
        "mssa_learned_projection": lambda x, _: _mssa_neural(x, learn_projection=True),
    }
    plans = ((multi_step, ForecastComparisonPlan((96, 108, 120), 4)),
             (page, ForecastComparisonPlan((95, 107, 119), 1)))
    provenance = comparison_provenance(
        {"train": SourceFingerprint.arrays("Airline TRAIN", train)},
        ("fedot-ind", "industrial-forecasting", "industrial-neural", "numpy", "sktime", "PyWavelets", "torch"),
    )
    rows = [row for models, plan in plans for row in forecast_rows(train, models, plan)]
    return comparison_report(
        "airline_method_study", provenance=provenance,
        protocol={"partition": "TRAIN", "excluded_tail": AIRLINE_EXCLUDED_TAIL, "time": "wall_clock"},
        results={"data": "Airline", "partition": "TRAIN-only (last 12 excluded)", "rows": rows},
    )


def _fedot_forecast(name):
    from dataclasses import replace
    from fedot import create_data
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    from fedot_ind.integration.fedot.extensions import industrial_extension_scope

    x = (np.arange(96) * .02 + np.sin(np.arange(96) / 3)).astype(np.float32)
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(4))
    with industrial_extension_scope():
        data = create_data(x, target=x, task=task, ts_forecast_horizon=4, without_target=True, use_cache=False)
        pipeline = PipelineBuilder().add_node(name, params={"forecast_length": 4}).build()
        pipeline.fit(data)
        return np.asarray(pipeline.predict(replace(data, target=None, state="predict")).predict).reshape(-1).tolist()


def compare_execution():
    from distributed import Client, LocalCluster

    names = ["wavelet_hfcm_forecaster", "arino_forecaster"]
    sequential = [_fedot_forecast(name) for name in names]
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            parallel = client.gather(client.map(_fedot_forecast, names))
    np.testing.assert_allclose(sequential, parallel, rtol=1e-6, atol=1e-6)
    return {"parity": True, "operations": names, "workers": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dask", action="store_true", help="Check sequential/worker parity on a synthetic series.")
    args = parser.parse_args()
    print(json.dumps(compare_execution() if args.dask else compare_on_airline_train(), indent=2))


if __name__ == "__main__":
    main()
