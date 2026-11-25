"""Matched forecasting controls; only fixed internal TRAIN origins are evaluated."""

import argparse
from functools import partial
import json

import numpy as np

from benchmark.industrial.experiments.tsf_comparison import (
    AIRLINE_EXCLUDED_TAIL, ForecastComparisonPlan, SourceFingerprint, comparison_provenance, comparison_report,
    forecast_rows, load_airline_train, measure,
)
from industrial_forecasting.dwt import ArinoSpec, predict_arino, predict_jin_kim
from industrial_forecasting.metrics import score_forecast, seasonal_repeat
from industrial_forecasting.neural.coordinate_forecast import fit_amortized_wire, predict_segment_wire
from industrial_forecasting.neural.mssa import MSSAForecaster
from industrial_forecasting.neural.page_projection import fit_page_projection
from industrial_forecasting.neural.wavelet_attention import WaveletAttentionSpec, fit_wavelet_attention
from industrial_forecasting.page_models import (
    fit_page_mssa, page_tensor, reconstruct_matrix_page, reconstruct_tensor_page,
)
from industrial_forecasting.wire import predict_positional_wire

AIRLINE_ORIGINS = (96, 108, 120)
WIRE_ORIGINS = (64, 76, 104, 132)
MACRO_ORIGINS = (95, 119, 143, 167)
MACRO_COLUMNS = ("realgdp", "realcons", "realinv")


def _arima(values, horizon, *, order, seasonal_order=(0, 0, 0, 0)):
    from statsmodels.tsa.arima.model import ARIMA

    return np.asarray(ARIMA(values, order=order, seasonal_order=seasonal_order).fit().forecast(horizon))


def _ann(values, horizon, *, lags=3):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    windows = np.lib.stride_tricks.sliding_window_view(values, lags + 1)
    scaler = StandardScaler().fit(values[:, None])
    normalized = (windows - scaler.mean_[0]) / scaler.scale_[0]
    fitted = MLPRegressor(hidden_layer_sizes=(3,), activation="tanh", solver="lbfgs",
                          random_state=0, max_iter=1000).fit(normalized[:, :-1], normalized[:, -1])
    state = normalized[-1, 1:].tolist()
    for step in range(horizon):
        state.append(float(fitted.predict(np.asarray(state[step:step + lags])[None])[0]))
    return np.asarray(state[-horizon:]) * scaler.scale_[0] + scaler.mean_[0]


def _garch_detail(values, horizon):
    from arch import arch_model

    # Positive conditional volatility is not a forecast of the signed detail.
    fitted = arch_model(values, mean="Zero", vol="GARCH", p=3, q=3).fit(disp="off")
    variance = np.asarray(fitted.forecast(horizon=horizon, reindex=False).variance)[-1]
    return np.zeros(horizon), variance, int(fitted.convergence_flag)


def _without_arino_dwt(x, h, trend, season):
    # Keep the same models, but replace DWT with a one-sided moving mean.
    size = 12
    history = np.cumsum(np.r_[0., x])
    moving_mean = np.asarray([(history[i + 1] - history[max(0, i + 1 - size)]) / min(i + 1, size)
                              for i in range(len(x))])
    return trend(moving_mean, h) + season(x - moving_mean, h)


def wavelet_comparison(train, *, with_garch=False):
    trend = partial(_arima, order=(1, 3, 0))
    season = partial(_arima, order=(0, 0, 0), seasonal_order=(0, 1, 1, 12))
    methods = {
        "arino_arima_dwt": lambda x, h: predict_arino(
            x, h, ArinoSpec(padding=8), padding_forecast=trend, trend_forecast=trend, seasonal_forecast=season,
        ),
        "same_arima_models_with_moving_mean": lambda x, h: _without_arino_dwt(x, h, trend, season),
    }
    for kind in ("joint", "uniform", "cnn", "lstm"):
        def cwt(x, h, mode=kind):
            spec = WaveletAttentionSpec(
                window=24, horizon=h, width=6, attention=mode == "joint",
                branch="joint" if mode == "uniform" else mode,
            )
            return fit_wavelet_attention((x,), spec, epochs=4, seed=2).predict(x)
        methods[f"cwt_{kind}"] = cwt
    result = list(forecast_rows(train, methods, ForecastComparisonPlan(AIRLINE_ORIGINS, 4)))
    ar = partial(_arima, order=(2, 1, 0))
    for horizon in (2, 4):
        methods = {"raw_arima": ar, "raw_ann": _ann}
        for label, forecast in (("arima", ar), ("ann", _ann)):
            methods[f"edge_{label}"] = partial(predict_jin_kim, first_end=80, approximation_forecast=forecast)
        methods["edge_ann_detail_ann"] = partial(
            predict_jin_kim, first_end=80, approximation_forecast=_ann, details=(1,), detail_forecast=_ann,
        )
        if with_garch:
            garch = []

            def mean(values, h):
                predicted, variance, status = _garch_detail(values, h)
                garch.append({"variance": variance.tolist(), "convergence_flag": status})
                return predicted

            methods["edge_ann_zero_mean_garch"] = partial(
                predict_jin_kim, first_end=80, approximation_forecast=_ann, details=(1,), detail_forecast=mean,
            )
        rows = list(forecast_rows(train, methods, ForecastComparisonPlan(AIRLINE_ORIGINS, horizon)))
        if with_garch:
            for row, diagnostic in zip((r for r in rows if r["model"].endswith("garch")), garch):
                row["garch"] = diagnostic
        result.extend(rows)
    return result


def _frequency_switch():
    t = np.arange(180.)
    return np.sin(np.where(t < 72, t / 3, 24 + (t - 72) / 7))


def wire_comparison(signal):
    import torch

    def amortized(x, horizon, *, localize):
        model = fit_amortized_wire((x,), context=24, horizon=horizon, width=10, epochs=6,
                                   seed=3, localize=localize)
        with torch.inference_mode():
            return model(torch.as_tensor(x[-24:][None], dtype=torch.float32)).numpy()[0]

    def positional(x, h, *, mode):
        return predict_positional_wire(x, h, mode=mode, lags=12, frequencies=(8., 16.), widths=1., centers=1.)

    methods = {
        "wire_position": partial(positional, mode="wire"),
        "periodic_position": partial(positional, mode="fourier"),
        "no_position": partial(positional, mode="none"),
        "segment_wire": lambda x, h: predict_segment_wire(x[-24:], h, width=10, epochs=6, seed=3).values,
        "amortized_wire": partial(amortized, localize=True),
        "amortized_periodic": partial(amortized, localize=False),
        "seasonal_repeat": lambda x, h: seasonal_repeat(x, h, 12),
    }
    return [row for h in (4, 12, 24) for row in forecast_rows(signal, methods, ForecastComparisonPlan(WIRE_ORIGINS, h))]


def _tensor_fixture():
    rng = np.random.default_rng(17)
    shape = (4, 6, 20)
    expected = np.einsum("ar,br,cr->abc", *(rng.normal(size=(size, 2)) for size in shape))
    mask = rng.random(shape) < .15
    return expected, mask


def tensor_comparison(expected, mask):
    shape = expected.shape
    observed = expected.copy()
    observed[mask] = np.nan
    series = observed.transpose(2, 1, 0).reshape(-1, shape[0])
    np.testing.assert_array_equal(page_tensor(series, shape[1]), observed)
    rows = []
    for name, estimate in (
        ("matrix_page", partial(reconstruct_matrix_page, length=shape[1], rank=2)),
        ("tensor_page", lambda x: reconstruct_tensor_page(x, length=shape[1], rank=2, seed=17).values),
    ):
        actual, cost = measure(estimate, series, trace_memory=True)
        rows.append({"model": name, "removed_count": int(mask.sum()), **cost,
                     "rmse": score_forecast("rmse", expected[mask], actual[mask]).value})
    return rows


def _macro_growth():
    from statsmodels.datasets import macrodata

    levels = macrodata.load_pandas().data.loc[:, list(MACRO_COLUMNS)].to_numpy()
    return np.diff(np.log(levels), axis=0)[:MACRO_ORIGINS[-1] + 1]


def multichannel_page_comparison(growth):
    import torch

    rows = []
    # All methods predict the same held-out block end. Scaling is fit on past rows.
    for train, evaluate in ForecastComparisonPlan(MACRO_ORIGINS, 1).windows(len(growth)):
        history, target = growth[train], growth[evaluate][0]
        mean, scale = history.mean(axis=0), history.std(axis=0)
        known = (history - mean) / scale
        available = known[-11:]

        def neural(learn):
            model = fit_page_projection(known, length=12, rank=2, learn_projection=learn, epochs=8)
            with torch.inference_mode():
                return model(torch.as_tensor(available.T.copy(), dtype=torch.float32)).numpy()

        methods = {
            "last_growth": lambda: available[-1],
            "page_reference": lambda: fit_page_mssa(known, length=12, rank=2).predict_block_end(available),
            "legacy_mssa_linear": lambda: MSSAForecaster(
                forecast_horizon=1, window_size=12, rank=2, lag_order=11, head_policy="linear", device="cpu",
            ).fit(known).predict().reshape(-1),
            "neural_frozen": partial(neural, False),
            "neural_learned": partial(neural, True),
        }
        for name, method in methods.items():
            prediction, cost = measure(method)
            forecast = np.asarray(prediction).reshape(-1) * scale + mean
            rows.append({"origin": evaluate.start, "columns": MACRO_COLUMNS, "model": name, **cost,
                         "rmse": score_forecast("rmse", target, forecast).value,
                         "absolute_error": np.abs(target - forecast).tolist()})
    return rows


def soft_dtw_comparison():
    import torch

    from fedot_ind.core.metrics.loss.soft_dtw_torch import soft_dtw_divergence

    time = torch.arange(30., dtype=torch.float64)
    reference = torch.exp(-((time - 10) / 2) ** 2)[None, :, None]
    rows = []
    for shift in (0, 1, 3, 5):
        prediction = torch.exp(-((time - 10 - shift) / 2) ** 2)[None, :, None]
        rows.append({"shift": shift, "mse": float((reference - prediction).square().mean()),
                     "soft_dtw_divergence": float(soft_dtw_divergence(reference, prediction, gamma=.02))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-garch", action="store_true", help="Requires the optional arch package.")
    args = parser.parse_args()
    train, frequency_shift, macro = load_airline_train(), _frequency_switch(), _macro_growth()
    expected, mask = _tensor_fixture()
    provenance = comparison_provenance(
        {"airline_train": SourceFingerprint.arrays("Airline TRAIN", train),
         "frequency_shift": SourceFingerprint.arrays("Synthetic frequency switch", frequency_shift),
         "masked_tensor": SourceFingerprint.arrays("Synthetic tensor and mask", expected, mask),
         "macro_growth": SourceFingerprint.arrays("statsmodels macrodata log differences", macro)},
        ("fedot-ind", "industrial-forecasting", "industrial-neural", "numpy", "scipy", "statsmodels",
         "scikit-learn", "sktime", "PyWavelets", "torch", "tensorly", "arch"),
    )
    report = comparison_report(
        "forecast_method_validation", provenance=provenance,
        protocol={"excluded_airline_tail": AIRLINE_EXCLUDED_TAIL, "with_garch": args.with_garch,
                  "tensor_seed": 17, "measurement": {"time": "wall_clock", "memory": "python_allocations"}},
        results={"airline_internal_train": wavelet_comparison(train, with_garch=args.with_garch),
                 "frequency_shift_synthetic": wire_comparison(frequency_shift),
                 "masked_tensor_synthetic": tensor_comparison(expected, mask),
                 "macrodata_multichannel": multichannel_page_comparison(macro),
                 "shifted_peak_synthetic": soft_dtw_comparison()},
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
